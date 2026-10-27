"""
What does a no-code run of run_hs_seq predict when it can see a treatment start, and when it cannot? (no training, forward passes, CPU)

A "start" is a new biologic: Biologic_ever goes up between visit k and the next visit k + 1. The treatment columns of the features
(TREATMENT_COLS) at visit k + 1 show it; at visit k they do not. Two readings of the model's prediction of y:
  unseen : the prediction at visit k, from its own features (the start is in the future of k and cannot be read)
  seen   : the prediction at visit k with the treatment columns of visit k replaced by those of visit k + 1 (an oracle edit: the model is
           shown the start, as if it were known at k). Everything else, and the history, stays as it was.
Part 1: one patient (default HS028540, cut at month 15.2): unseen, seen, and the two natural predictions at visit k + 1 (real features;
        and the treatment columns put back to those of visit k, "start hidden").
Part 2: all test patients with a labelled visit k before a biologic start: the mean shift (seen - unseen) and the RMSE of both against y(k).
The edit is a statement about the model's learned association between these features and y, not an estimate of a treatment effect: scores
fall after starts also for reasons the model cannot separate (flares prompt the start, surgeries, regression to the mean).

    python -m experiments.latent.counterfactual_treatment runs/hs_seq_<stamp> [more runs] --cpu
"""
import argparse

import numpy as np
import torch

from experiments.latent import run_hs_seq as R
from experiments.latent.run import build_model

TREATMENT_COLS = ["n_tx_started", "days_since_tx_start", "n_tx_ended", "n_tx_active", "Biologic_ever", "Biologic_active", "max_bio_line"]


@torch.no_grad()
def predict_at(model, data, i, k, device, edit=None):
    """Decoded mean (original units) at row k of patient i from the first k + 1 rows; edit = (columns, source row): those columns of row k
    take the values (and masks) of the source row, for this call only."""
    j = data.off[i] + k
    if edit is not None:
        cols, src = edit
        keep_x, keep_m = data.X[j, cols].copy(), data.M[j, cols].copy()
        data.X[j, cols], data.M[j, cols] = data.X[data.off[i] + src, cols], data.M[data.off[i] + src, cols]
    try:
        code = torch.zeros(1, 0, device=device)
        batch = data.pack([(int(i), k + 1, None)], code, device)
        model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
        m = model.posterior(batch["t"][:, -1:])[0][:, 0]
        return float(model.decoder(m, code).mean[0, 0] * data.ystd + data.ymean)
    finally:
        if edit is not None:
            data.X[j, cols], data.M[j, cols] = keep_x, keep_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--patient", default="HS028540")
    ap.add_argument("--cut_month", type=float, default=15.2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    data = R.Data(R.LIGHT)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    cols = [data.names.index(c) for c in TREATMENT_COLS]
    ever = data.names.index("Biologic_ever")

    # ---- the biologic starts of the test patients: visit k with a labelled y, the next visit shows a new biologic
    events = []
    for i in data.test_idx:
        a, b = data.off[i], data.off[i + 1]
        e = np.where(data.M[a:b, ever], data.X[a:b, ever], np.nan)  # standardized, but monotone: an increase is an increase
        up = np.flatnonzero(np.diff(e) > 1e-6)
        events += [(int(i), int(k)) for k in up if data.ym[a + k]]
    print(f"{len(events)} biologic starts of the test patients with a labelled visit just before the start")
    pid = list(data.pid).index(args.patient)
    k_cut = int(np.argmin(np.abs(data.months[data.off[pid]:data.off[pid + 1]] - args.cut_month)))
    k_after = next(k for k in range(k_cut + 1, data.n_rows(pid)) if data.X[data.off[pid] + k, ever] > data.X[data.off[pid] + k_cut, ever] + 1e-6)
    t = data.months[data.off[pid]:data.off[pid + 1]]
    own = np.where(data.raw["own_score"][1][data.off[pid]:data.off[pid] + data.n_rows(pid)], data.raw["own_score"][0][data.off[pid]:data.off[pid] + data.n_rows(pid)], np.nan)
    w = (t > t[k_after]) & (t <= t[k_after] + 6.0) & np.isfinite(own)
    true_after = own[w].mean() if w.any() else np.nan
    y_cut = data.y[data.off[pid] + k_cut]

    for run in args.runs:
        model = build_model(R.CFG, None, 1, None).to(device)
        model.load_state_dict(torch.load(f"{run}/model.pt", map_location=device)["model"])
        model.eval()
        print(f"\n===== {run}")
        print(f"Part 1: {args.patient}, cut visit at month {t[k_cut]:.1f}, the biologic start is read at visit month {t[k_after]:.1f}")
        un = predict_at(model, data, pid, k_cut, device)
        se = predict_at(model, data, pid, k_cut, device, edit=(cols, k_after))
        nat = predict_at(model, data, pid, k_after, device)
        hid = predict_at(model, data, pid, k_after, device, edit=(cols, k_cut))
        print(f"  at the cut (month {t[k_cut]:.1f}), true next-6-month mean {y_cut:.1f}:  start unseen {un:.1f}   start shown {se:.1f}   (shift {se - un:+.1f})")
        print(f"  at month {t[k_after]:.1f}, true next-6-month mean {true_after:.1f}:  start visible (real) {nat:.1f}   start hidden {hid:.1f}   (shift {nat - hid:+.1f})")
        un_all, se_all, y_all = [], [], []
        for i, k in events:
            un_all.append(predict_at(model, data, i, k, device))
            se_all.append(predict_at(model, data, i, k, device, edit=(cols, k + 1)))
            y_all.append(data.y[data.off[i] + k])
        un_all, se_all, y_all = map(np.array, (un_all, se_all, y_all))
        shift = se_all - un_all
        rmse = lambda p: np.sqrt(np.mean((p - y_all) ** 2))
        print(f"Part 2: {len(events)} starts.  RMSE vs y(k):  unseen {rmse(un_all):.3f}   shown {rmse(se_all):.3f}")
        print(f"  shift (shown - unseen): mean {shift.mean():+.2f}  median {np.median(shift):+.2f}  share negative {np.mean(shift < 0):.2f}  "
              f"[10%, 90%] = [{np.percentile(shift, 10):+.2f}, {np.percentile(shift, 90):+.2f}]")
        print(f"  true y(k) mean {y_all.mean():.2f};  mean prediction unseen {un_all.mean():.2f}, shown {se_all.mean():.2f}")


if __name__ == "__main__":
    main()
