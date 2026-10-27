"""
The past trajectory and the forecast at a cut, for random test patients (no training, forward passes on the CPU).

For each patient a random labelled visit is the cut t. Shown: the scores up to the cut (grey: the history the posterior has read), the
predictions the model made at the earlier visits (faint blue; each one was made with the data up to that visit), the prediction made at
the cut for the mean score of (t, t + 6 months] (blue segment, +- 1 decoder std), the true mean of that window (black segment, the target)
and the scores after the cut (black circles: what actually happened, never read by the model).

    python -m experiments.latent.plot_cuts runs/hs_seq_<stamp> --dim_z 32 --dec_hidden 128 128 [--extra_features] --cpu
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.plot_rows import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, states
from experiments.latent.run import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--extra_features", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min_rows", type=int, default=20)
    ap.add_argument("--min_history", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    data = R.Data(R.LIGHT, extra=args.extra_features)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    model = build_model(R.CFG, None, 1, None).to(device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device)["model"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    pool = [i for i in data.test_idx if data.ym[data.off[i]:data.off[i + 1]].sum() >= args.min_rows]
    pick = rng.choice(pool, args.n, replace=False)

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "font.size": 9})
    cols = 5
    fig, axes = plt.subplots(-(-args.n // cols), cols, figsize=(4.0 * cols, 3.1 * -(-args.n // cols)))
    for ax, i in zip(axes.ravel(), pick):
        s = slice(data.off[i], data.off[i + 1])
        t, ok = data.months[s], data.ym[s]
        mean, std = states(model, data, i, device)
        own = np.where(data.raw["own_score"][1][s], data.raw["own_score"][0][s], np.nan)
        cand = np.flatnonzero(ok & (np.arange(len(t)) >= args.min_history))
        k = int(rng.choice(cand))
        tc, y = t[k], data.y[s][k]
        past, future = np.arange(len(t)) <= k, np.arange(len(t)) > k
        win = future & (t <= tc + 6.0)
        ax.axvspan(tc, tc + 6.0, color=BLUE, alpha=0.06, linewidth=0, zorder=0)
        ax.axvline(tc, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.plot(t[past], own[past], color=MUTED, lw=0.8, alpha=0.7, zorder=2)
        ax.scatter(t[past], own[past], s=12, color=MUTED, linewidths=0, label="scores up to the cut (read by the model)", zorder=3)
        ax.plot(t[:k + 1], mean[:k + 1], color=BLUE, lw=1.0, alpha=0.45, label="earlier predictions (made at each visit)", zorder=3)
        ax.scatter(t[future & (t <= tc + 12.0)], own[future & (t <= tc + 12.0)], s=26, facecolor="none", edgecolor=INK, linewidths=1.0,
                   label="scores after the cut (not read)", zorder=4)
        ax.fill_between([tc, tc + 6.0], mean[k] - std[k], mean[k] + std[k], color=BLUE, alpha=0.16, linewidth=0, zorder=2)
        ax.plot([tc, tc + 6.0], [mean[k]] * 2, color=BLUE, lw=3.0, solid_capstyle="butt", label="prediction for the next 6 months", zorder=6)
        ax.plot([tc, tc + 6.0], [y] * 2, color=INK, lw=3.0, solid_capstyle="butt", label="true mean of the next 6 months", zorder=5)
        ax.set_xlim(max(0.0, tc - 36.0), tc + 12.0)
        ax.set_title(f"{data.pid[i]}  cut at month {tc:.0f}:  predicted {mean[k]:.1f}  true {y:.1f}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("month")
        ax.grid(axis="y", color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = f"{args.run}/past_and_forecast.png"
    fig.savefig(out, dpi=105)
    print("saved", out)


if __name__ == "__main__":
    main()
