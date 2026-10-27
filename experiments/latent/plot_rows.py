"""
Posterior prediction against the true target for test patients, from a no-code run of run_hs_seq (no training, forward passes only).

For each patient: the decoded posterior mean at every visit (one pass of the forward posterior, so the value at a visit reads only the
visits up to it) with a band of +- 1 decoder std, the true target y(t) (the mean score in (t, t + 6 months], so it leads the scores),
the past 6-month mean feature as the simple baseline, and the raw scores. The last visit carries the official label (y_is_real = 1).
The patients are random test patients (seed) with at least `min_rows` labelled visits: none picked by their score.

    python -m experiments.latent.plot_rows runs/hs_seq_<stamp> --dim_z 32 --dec_hidden 128 128 [--extra_features]
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.run import build_model

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette


@torch.no_grad()
def states(model, data, i, device):
    """Decoded mean and std (original units) at every visit of patient i, code of width 0."""
    n = data.n_rows(i)
    code = torch.zeros(1, 0, device=device)
    batch = data.pack([(int(i), n, None)], code, device)
    model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
    m = model.posterior(batch["t"])[0]
    dist = model.decoder(m, code[:, None].expand(-1, m.shape[1], -1))
    return (dist.mean[0, :n, 0] * data.ystd + data.ymean).cpu().numpy(), (dist.stddev[0, :n, 0] * data.ystd).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--extra_features", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min_rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force the CPU (the plot needs 20 forward passes only)")
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
        t, ok, real = data.months[s], data.ym[s], data.yr[s]
        mean, std = states(model, data, i, device)
        y = np.where(ok, data.y[s], np.nan)
        base = np.where(data.raw["mean_6m"][1][s], data.raw["mean_6m"][0][s], np.nan)
        own = np.where(data.raw["own_score"][1][s], data.raw["own_score"][0][s], np.nan)
        ax.scatter(t, own, s=7, color=MUTED, alpha=0.35, linewidths=0, label="scores", zorder=1)
        ax.plot(t, base, color=MUTED, lw=1.0, ls=":", label="past 6-month mean", zorder=2)
        ax.fill_between(t, mean - std, mean + std, color=BLUE, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(t, mean, color=BLUE, lw=1.6, label="posterior prediction (at the visit)", zorder=4)
        ax.plot(t, y, color=INK, lw=1.4, label="true target (next 6 months)", zorder=3)
        if (ok & real).any():
            k = np.flatnonzero(ok & real)[0]
            ax.scatter([t[k]], [y[k]], s=46, marker="*", color=ORANGE, edgecolor=SURFACE, linewidths=0.8, zorder=6,
                       label="official label (last visit)")
        e = lambda p: np.sqrt(np.nanmean((p[ok] - data.y[s][ok]) ** 2))
        ax.set_title(f"{data.pid[i]}   RMSE model {e(mean):.2f} | 6-month mean {np.sqrt(np.nanmean((base[ok] - data.y[s][ok]) ** 2)):.2f}",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("month")
        ax.grid(axis="y", color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = f"{args.run}/posterior_vs_true.png"
    fig.savefig(out, dpi=105)
    print("saved", out)


if __name__ == "__main__":
    main()
