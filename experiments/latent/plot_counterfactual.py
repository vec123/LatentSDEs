"""
Figure of the treatment-start comparison of counterfactual_treatment.py (no training, CPU), one row per run.
  left   : HS028540 around its cut: scores, treatments and surgeries (from the public csv files), the model's predictions at each visit
           (blue line: made at the visit, the future not read), and at the cut the prediction for the next 6 months with the start unseen
           (blue), with the start shown (orange, treatment columns of the next visit edited in), and the true mean (black); at the next visit,
           the real prediction (the start visible, orange diamond) and the same visit with the start hidden (blue diamond).
  middle : the shift of the prediction (shown - unseen) over all biologic starts of the test patients that follow a labelled visit.
  right  : calibration of the two predictions on those visits: the mean true y in deciles of the prediction (the diagonal is perfect).
The edit is the model's learned association between the treatment columns and y, not a treatment effect.

    python -m experiments.latent.plot_counterfactual runs/hs_seq_<stamp> [more runs] --cpu
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.counterfactual_treatment import TREATMENT_COLS, predict_at
from experiments.latent.plot_rows import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, states
from experiments.latent.run import build_model

PUBLIC = "test_data/hs_public_dataset/public/train"


def events_of(data):
    ever = data.names.index("Biologic_ever")
    out = []
    for i in data.test_idx:
        a, b = data.off[i], data.off[i + 1]
        e = np.where(data.M[a:b, ever], data.X[a:b, ever], np.nan)
        out += [(int(i), int(k)) for k in np.flatnonzero(np.diff(e) > 1e-6) if data.ym[a + k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--patient", default="HS028540")
    ap.add_argument("--cut_month", type=float, default=15.2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--xgb", help="runs/xgb/edit_cache.npz of xgb_edit.py: adds a row for XGBoost")
    ap.add_argument("--out", default="runs/treatment_start_comparison.png")
    args = ap.parse_args()
    labels = args.labels or [r.split("/")[-1] for r in args.runs]
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    data = R.Data(R.LIGHT)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    cols = [data.names.index(c) for c in TREATMENT_COLS]
    ever = data.names.index("Biologic_ever")
    events = events_of(data)

    pid = list(data.pid).index(args.patient)
    a, n = data.off[pid], data.n_rows(pid)
    t = data.months[a:a + n]
    own = np.where(data.raw["own_score"][1][a:a + n], data.raw["own_score"][0][a:a + n], np.nan)
    k_cut = int(np.argmin(np.abs(t - args.cut_month)))
    k_after = next(k for k in range(k_cut + 1, n) if data.X[a + k, ever] > data.X[a + k_cut, ever] + 1e-6)
    y_cut = data.y[a + k_cut]
    v = pd.read_csv(f"{PUBLIC}/visits.csv")
    d0 = pd.to_datetime(v[v.patient_id == args.patient].visit_date).min()
    month = lambda s: (pd.to_datetime(s) - d0).dt.days / 30.4375
    tx = pd.read_csv(f"{PUBLIC}/treatments.csv")
    tx = tx[tx.patient_id == args.patient].assign(start=lambda d: month(d.start_date), end=lambda d: month(d.end_date))
    sg = pd.read_csv(f"{PUBLIC}/surgeries.csv")
    sg = sg[sg.patient_id == args.patient].assign(month=lambda d: month(d.procedure_date))

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "font.size": 9.5})
    import os
    results = []  # per model: (label, unseen, shown, next visit real, next visit hidden, per-visit predictions, arrays over the starts)
    for run, label in zip(args.runs, labels):
        cache = f"{run}/edit_cache.npz"
        if os.path.exists(cache):  # a run is computed once
            c = np.load(cache)
            results.append((label, *(float(c[k]) for k in ("un", "se", "nat", "hid")), c["factual"], c["un_all"], c["se_all"], c["y_all"]))
            continue
        model = build_model(R.CFG, None, 1, None).to(device)
        model.load_state_dict(torch.load(f"{run}/model.pt", map_location=device)["model"])
        model.eval()
        un = predict_at(model, data, pid, k_cut, device)
        se = predict_at(model, data, pid, k_cut, device, edit=(cols, k_after))
        nat = predict_at(model, data, pid, k_after, device)
        hid = predict_at(model, data, pid, k_after, device, edit=(cols, k_cut))
        factual, _ = states(model, data, pid, device)
        un_all, se_all, y_all = [], [], []
        for i, k in events:
            un_all.append(predict_at(model, data, i, k, device))
            se_all.append(predict_at(model, data, i, k, device, edit=(cols, k + 1)))
            y_all.append(data.y[data.off[i] + k])
        un_all, se_all, y_all = map(np.array, (un_all, se_all, y_all))
        np.savez(cache, un=un, se=se, nat=nat, hid=hid, factual=factual, un_all=un_all, se_all=se_all, y_all=y_all)
        results.append((label, un, se, nat, hid, factual, un_all, se_all, y_all))
        print(f"{label}: computed", flush=True)
    if args.xgb:  # runs/xgb/edit_cache.npz of xgb_edit.py: the same quantities for XGBoost
        c = np.load(args.xgb)
        assert int(c["k_cut"]) == k_cut and int(c["k_after"]) == k_after and len(c["un"]) == len(events), "the XGBoost cache is for other starts"
        un_x, se_x, nat_x, hid_x, _ = c["pat"]
        results.append(("XGBoost (same 89 features)", un_x, se_x, nat_x, hid_x, c["factual"], c["un"], c["se"], c["y"]))
    fig, axes = plt.subplots(len(results), 3, figsize=(17, 4.9 * len(results)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for r, (label, un, se, nat, hid, factual, un_all, se_all, y_all) in enumerate(results):
        # ---- patient panel
        ax = axes[r, 0]
        lo, hi = args.cut_month - 8.0, args.cut_month + 9.0
        top = np.nanmax(own[(t > lo) & (t < hi)]) + 4.0
        for _, x in tx.iterrows():  # treatment bars under the scores
            end = x.end if np.isfinite(x.end) else hi
            bio = x.treatment_class == "Biologic"
            ax.plot([x.start, end], [0.6 if bio else 1.6] * 2, color=ORANGE if bio else MUTED, lw=5 if bio else 3, alpha=0.9,
                    solid_capstyle="butt", zorder=2)
            if bio:
                ax.text(max(x.start, lo) + 0.1, 1.1, x.drug_name, fontsize=7.5, color=INK, va="bottom")
        ax.scatter(sg.month, [2.9] * len(sg), marker="v", s=40, color=INK, zorder=3)
        ax.text(sg.month.min(), 3.3, "surgeries", fontsize=7.5, color=INK, ha="left")
        ax.axvline(t[k_cut], color=MUTED, lw=0.9, ls="--")
        ax.plot(t, own, color=MUTED, lw=0.8, alpha=0.7)
        ax.scatter(t, own, s=14, color=MUTED, linewidths=0, label="scores", zorder=3)
        ax.plot(t, factual, color=BLUE, lw=1.0, alpha=0.5, label="prediction made at each visit")
        w = 6.0
        ax.plot([t[k_cut], t[k_cut] + w], [un] * 2, color=BLUE, lw=3.2, solid_capstyle="butt", label="at the cut: start unseen", zorder=6)
        ax.plot([t[k_cut], t[k_cut] + w], [se] * 2, color=ORANGE, lw=3.2, solid_capstyle="butt", label="at the cut: start shown (edited in)", zorder=6)
        ax.plot([t[k_cut], t[k_cut] + w], [y_cut] * 2, color=INK, lw=3.2, solid_capstyle="butt", label="true mean of the next 6 months", zorder=5)
        ax.scatter([t[k_after]], [nat], marker="D", s=44, color=ORANGE, edgecolor=SURFACE, zorder=7, label="next visit: start visible (real)")
        ax.scatter([t[k_after]], [hid], marker="D", s=44, color=BLUE, edgecolor=SURFACE, zorder=7, label="next visit: start hidden")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, top)
        ax.set_xlabel("month")
        ax.set_title(f"{label}   {args.patient}: cut {un:.1f} unseen -> {se:.1f} shown (true {y_cut:.1f})", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # ---- shift histogram
        ax = axes[r, 1]
        shift = se_all - un_all
        ax.hist(shift, bins=40, color=ORANGE, alpha=0.85)
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.axvline(np.median(shift), color=INK, lw=1.4, ls="--")
        ax.set_title(f"shift (shown - unseen), {len(shift)} biologic starts\nmedian {np.median(shift):+.2f}, {np.mean(shift < 0):.0%} negative",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("change of the predicted next-6-month mean")
        ax.set_ylabel("starts")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # ---- calibration by decile of the prediction
        ax = axes[r, 2]
        for pred, col, name in ((un_all, BLUE, "start unseen"), (se_all, ORANGE, "start shown")):
            edges = np.quantile(pred, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, 9)
            xs = [pred[b == j].mean() for j in range(10)]
            ys = [y_all[b == j].mean() for j in range(10)]
            ax.plot(xs, ys, "o-", color=col, lw=1.6, ms=5, label=f"{name}: RMSE {np.sqrt(np.mean((pred - y_all) ** 2)):.2f}")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=MUTED, lw=0.9, ls=":")
        ax.set_xlabel("mean prediction in the decile")
        ax.set_ylabel("mean true y")
        ax.set_title("calibration at the visits before a start", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        axes[r, 0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=105)
    print("saved", args.out)


if __name__ == "__main__":
    main()
