"""
Leak check, no training. Does the per-patient code table carry information about the future of its training patient?

300 training patients and 300 test patients, each scored at the last visit (the row with the targets.csv label, y_is_real = 1),
with the same step rule as run_hs_seq.evaluate (history up to the visit, labels whose window ended 6 months before it). Four predictions:
  train, table code   the code the model learned for that patient during training (it saw the whole series, future included)
  train, fitted code  a code fitted on the history only, as for a test patient
  test, fitted code   the same for test patients (never trained on)
  code 0              the mean training code (no information)
If "train, table code" is far better than "train, fitted code", the table codes hold information that a history-only code
cannot give: the model learned to read it. If "train, fitted" is much better than "test, fitted", the GRU and decoder
themselves overfit the training patients.

    python -m experiments.latent.leak_check runs/hs_seq_<stamp> [--dim_z 32 --dec_hidden 128 128]
"""
import argparse

import numpy as np
import torch

from experiments.latent import run_hs_seq as R
from experiments.latent.compare_baselines import metrics
from experiments.latent.run import build_model


def real_rows(data, idxs, n, rng):
    rows = []
    for i in rng.permutation(idxs):
        s = slice(data.off[i], data.off[i + 1])
        last = np.flatnonzero(data.yr[s] & data.ym[s])
        if len(last) and last[0] >= 2:  # a history of 3 rows or more
            rows.append((int(i), int(last[0])))
        if len(rows) == n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=R.CFG["dim_z"])
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=R.CFG["decoder"]["hidden"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    data = R.Data(R.LIGHT)
    R.CFG["dim_feat"] = data.X.shape[1]
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    ck = torch.load(f"{args.run}/model.pt", map_location=device)
    model = build_model(R.CFG, None, 1, None).to(device)
    model.load_state_dict(ck["model"])
    table = ck["codes"].to(device)
    default = table.mean(0)
    pos = {int(i): k for k, i in enumerate(data.train_idx)}
    out = {}
    for group, idxs in (("train", data.train_idx), ("test", data.test_idx)):
        rows = real_rows(data, idxs, args.n, rng)
        items = [(i, k + 1, data.months[data.off[i] + k] - 6.0) for i, k in rows]
        true = np.array([data.y[data.off[i] + k] for i, k in rows])
        idx = np.array([data.off[i] + k for i, k in rows])
        fitted, tabled, zero = [], [], []
        for a in range(0, len(items), args.chunk):
            part = items[a:a + args.chunk]
            c, ok = R.fit_codes(model, data, part, device)
            c = torch.where(ok[:, None], c, default.expand_as(c))
            fitted.append(R.predict(model, data, part, c, device))
            zero.append(R.predict(model, data, part, default.expand_as(c).clone(), device))
            if group == "train":
                tc = table[torch.as_tensor([pos[i] for i, _, _ in part], device=device)]
                tabled.append(R.predict(model, data, part, tc, device))
            import time
            time.sleep(args.pause)
        last = np.where(data.raw["last_score"][1][idx], data.raw["last_score"][0][idx], data.train_mean_y)
        preds = {"persistence: last score": last, "mean code (no information)": np.concatenate(zero),
                 f"{group}, fitted code (history only)": np.concatenate(fitted)}
        if tabled:
            preds[f"{group}, table code (saw the future)"] = np.concatenate(tabled)
        out[group] = {n: metrics(p, true) for n, p in preds.items()}
        print(f"\n{group} patients: {len(rows)} last visits, y mean {true.mean():.2f} std {true.std():.2f}", flush=True)
        for n, m in out[group].items():
            print(f"  {n:38s} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
