"""
Do the patient-to-patient differences in the model's response to a biologic start show up in the data? (no training, CPU)

For each biologic start of the test patients that follows a labelled visit k (see counterfactual_treatment.py): unseen = the model's prediction at
visit k; shown = the prediction with the treatment columns of visit k + 1 edited in (the model is shown the start). shift = shown - unseen is the
model's predicted response of that patient. realized = y(k) - unseen is what happened beyond the prediction made without the start (y(k) already
contains the effect of the start). If the model sees real differences in response, patients with a larger predicted shift show a larger realized
difference. Reported: the correlation (with a bootstrap 95% interval over patients), the slope, and the deciles of the shift.
This is an association: starts follow flares, so a high-state patient is both predicted a large shift and likely to fall (regression to the mean).
As a control the script also reports the same correlation with the unseen prediction itself as the "predictor": if that correlates as strongly,
the shift only carries the level of the patient.

    python -m experiments.latent.eval_edit_heterogeneity runs/hs_seq_<stamp> [more runs] --cpu
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.counterfactual_treatment import TREATMENT_COLS, predict_at
from experiments.latent.plot_rows import BLUE, GRID, INK, MUTED, ORANGE, SURFACE
from experiments.latent.run import build_model


def corr_ci(a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.corrcoef(a, b)[0, 1]
    boots = []
    for _ in range(n_boot):
        j = rng.integers(0, len(a), len(a))
        boots.append(np.corrcoef(a[j], b[j])[0, 1])
    return r, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--out", default="runs/edit_heterogeneity.png")
    args = ap.parse_args()
    labels = args.labels or [r.split("/")[-1] for r in args.runs]
    data = R.Data(R.LIGHT)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    cols = [data.names.index(c) for c in TREATMENT_COLS]
    ever = data.names.index("Biologic_ever")
    events = []
    for i in data.test_idx:
        a, b = data.off[i], data.off[i + 1]
        e = np.where(data.M[a:b, ever], data.X[a:b, ever], np.nan)
        events += [(int(i), int(k)) for k in np.flatnonzero(np.diff(e) > 1e-6) if data.ym[a + k]]
    y = np.array([data.y[data.off[i] + k] for i, k in events])
    print(f"{len(events)} biologic starts", flush=True)

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 9.5})
    fig, axes = plt.subplots(1, len(args.runs), figsize=(6.2 * len(args.runs), 5.0), squeeze=False)
    for ax, run, label in zip(axes[0], args.runs, labels):
        model = build_model(R.CFG, None, 1, None)
        model.load_state_dict(torch.load(f"{run}/model.pt", map_location="cpu")["model"])
        model.eval()
        un = np.array([predict_at(model, data, i, k, "cpu") for i, k in events])
        se = np.array([predict_at(model, data, i, k, "cpu", edit=(cols, k + 1)) for i, k in events])
        shift, realized = se - un, y - un
        r, lo, hi = corr_ci(shift, realized)
        r0, lo0, hi0 = corr_ci(un, realized)
        slope = np.polyfit(shift, realized, 1)[0]
        print(f"\n===== {label}")
        print(f"corr(predicted shift, realized difference) = {r:+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]  slope {slope:+.2f}")
        print(f"control: corr(unseen prediction, realized difference) = {r0:+.3f}  [{lo0:+.3f}, {hi0:+.3f}]")
        # partial: the shift after taking out its linear dependence on the unseen prediction (the level of the patient)
        res_shift = shift - np.polyval(np.polyfit(un, shift, 1), un)
        res_real = realized - np.polyval(np.polyfit(un, realized, 1), un)
        rp, lop, hip = corr_ci(res_shift, res_real)
        print(f"corr(shift, realized) with the level (unseen prediction) taken out of both = {rp:+.3f}  [{lop:+.3f}, {hip:+.3f}]")
        edges = np.quantile(shift, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, shift, side="right") - 1, 0, 9)
        xs = [shift[b == j].mean() for j in range(10)]
        ys = [realized[b == j].mean() for j in range(10)]
        print("deciles of the shift (mean shift -> mean realized difference):", ", ".join(f"{a_:+.2f}->{b_:+.2f}" for a_, b_ in zip(xs, ys)))
        ax.plot(xs, ys, "o-", color=ORANGE, lw=1.8, ms=5, label=f"deciles of the predicted shift (r = {r:+.2f})")
        lim = [min(min(xs), min(ys)), max(max(xs), max(ys))]
        ax.plot(lim, lim, color=MUTED, ls=":", lw=0.9, label="realized = predicted")
        ax.set_xlabel("predicted shift (shown - unseen)")
        ax.set_ylabel("realized difference (y - unseen prediction)")
        ax.set_title(f"{label}: partial r (level removed) = {rp:+.2f} [{lop:+.2f}, {hip:+.2f}]", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=105)
    print("saved", args.out)


if __name__ == "__main__":
    main()
