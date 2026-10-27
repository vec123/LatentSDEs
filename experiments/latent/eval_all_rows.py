"""
Scores a run_hs_seq model on ALL labelled rows of the test patients (110,977 rows, y_mask = 1) in one pass, with the code set
to 0 (no per-patient fit): the table of the XGBoost comparison. One forward pass of the posterior over each test patient gives its
state at every visit; the decoder mean at the visit is the prediction. The posterior is forward only, so the state at a visit reads
the visits up to it; check_causality compares the prediction at a visit from the whole sequence with the one from the visits up to
it alone (they must agree: nothing later is read).

    python -m experiments.latent.eval_all_rows runs/hs_seq_<stamp> --dim_z 32 --dec_hidden 128 128
"""
import argparse
import time

import numpy as np
import torch
import yaml

from experiments.latent import run_hs_seq as R
from experiments.latent.compare_baselines import metrics
from experiments.latent.run import build_model


@torch.no_grad()
def predict_rows(model, data, patients, device, prefix=None, chunk=25, pause=0.0, dim_mu=0):
    """
    Decoded posterior mean at every row of each patient (a list of arrays, original units), code 0. prefix: a list of row counts;
    each patient is then given only its first n rows (used by the causality check).
    """
    model.eval()
    order = np.argsort([data.n_rows(i) if prefix is None else prefix[j] for j, i in enumerate(patients)])
    out = [None] * len(patients)
    for a in range(0, len(order), chunk):
        sel = order[a:a + chunk]
        items = [(int(patients[j]), data.n_rows(patients[j]) if prefix is None else int(prefix[j]), None) for j in sel]
        code = torch.zeros(len(items), dim_mu, device=device)  # width of the code the model was built with
        batch = data.pack(items, code, device)
        model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
        m = model.posterior(batch["t"])[0]  # (B, N, dim_z): the state at every row time
        mean = model.decoder(m, code[:, None].expand(-1, m.shape[1], -1)).mean[..., 0]
        mean = (mean * data.ystd + data.ymean).cpu().numpy()
        for b, j in enumerate(sel):
            out[j] = mean[b, :items[b][1]]
        time.sleep(pause)
    return out


def score_all(model, data, device, out_dir, pause=0.0, dim_mu=0):
    """The causality check and the scores on all labelled rows of the test patients (code 0); writes metrics_all_rows.yaml."""
    patients = data.test_idx
    rng = np.random.default_rng(0)
    pick = rng.choice(len(patients), 60, replace=False)
    sub = [patients[j] for j in pick]
    ks = [int(rng.integers(0, data.n_rows(i))) for i in sub]
    full = predict_rows(model, data, sub, device, dim_mu=dim_mu)
    short = predict_rows(model, data, sub, device, prefix=[k + 1 for k in ks], dim_mu=dim_mu)
    diff = max(abs(full[j][k] - short[j][k]) for j, k in enumerate(ks))
    print(f"causality check: max |prediction from the whole sequence - prediction from the rows up to it| = {diff:.2e} "
          f"(60 patients, one row each)", flush=True)
    preds = predict_rows(model, data, patients, device, pause=pause, dim_mu=dim_mu)
    P, Y, LAST, M6, real = [], [], [], [], []
    for i, p in zip(patients, preds):
        s = slice(data.off[i], data.off[i + 1])
        ok = data.ym[s]
        P.append(p[ok])
        Y.append(data.y[s][ok])
        LAST.append(np.where(data.raw["last_score"][1][s], data.raw["last_score"][0][s], data.train_mean_y)[ok])
        M6.append(np.where(data.raw["mean_6m"][1][s], data.raw["mean_6m"][0][s], data.train_mean_y)[ok])
        real.append(data.yr[s][ok])
    P, Y, LAST, M6, real = map(np.concatenate, (P, Y, LAST, M6, real))
    out = {}
    for name, sel in (("all rows", np.ones_like(real)), ("real rows (targets.csv)", real), ("derived rows", ~real)):
        out[name] = {"rows": int(sel.sum())}
        print(f"\n{name}: {sel.sum()} rows, y mean {Y[sel].mean():.2f} std {Y[sel].std():.2f}")
        for n, p in (("Train mean", np.full(sel.sum(), data.train_mean_y)), ("Persistence: last score", LAST[sel]),
                     ("Persistence: past 6-month mean (feature)", M6[sel]), ("Latent SDE (code 0)", P[sel])):
            m = metrics(p, Y[sel])
            out[name][n] = m
            print(f"  {n:42s} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}", flush=True)
    out["causality_max_diff"] = float(diff)
    yaml.safe_dump(out, open(f"{out_dir}/metrics_all_rows.yaml", "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=R.CFG["dim_z"])
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=R.CFG["decoder"]["hidden"])
    ap.add_argument("--data", default=R.LIGHT)
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--no_code", action="store_true")
    ap.add_argument("--extra_features", action="store_true")
    ap.add_argument("--treatment_u", action="store_true")
    ap.add_argument("--drop_engineered", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = R.Data(args.data, extra=args.extra_features, treatment_u=args.treatment_u, drop_engineered=args.drop_engineered)
    R.CFG["dim_feat"] = data.X.shape[1]
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    if args.no_code:
        R.CFG["dim_mu"] = 0
    if args.treatment_u:
        R.CFG["dim_u"] = data.U.shape[1]
    model = build_model(R.CFG, None, 1, None).to(device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device)["model"])
    score_all(model, data, device, args.run, pause=args.pause, dim_mu=R.CFG["dim_mu"])


if __name__ == "__main__":
    main()
