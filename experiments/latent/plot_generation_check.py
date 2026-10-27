"""
Figure: the generation check of diffusion_calibration.py as scatter plots (no computation; reads <run>/gen_check_s0_<s0>.npz, k = 0).
Per patient (100 training patients, the scheduled visits of the 36 months after the start at month 12): the level (mean score) and the
oscillation (mean |change| between successive scheduled visits) of the generated courses (drift + drawn flares + visit noise, no diffusion
of z; mean of 5 courses) against the data; left the start state drawn from the posterior, right the posterior mean.

    python -m experiments.latent.plot_generation_check runs/hs_fused_sde_<stamp>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.latent.plot_code_archetypes import GRID, INK, INK2, MUTED, SERIES, SURFACE  # noqa: E402


def main():
    run = sys.argv[1]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.4))
    for c, (s0, title) in enumerate((("1", "start state drawn from the posterior"), ("0", "start state = the posterior mean"))):
        f = np.load(f"{run}/gen_check_s0_{s0}.npz")
        D, G = f["data"], f[f"k_{list(f['ks']).index(0.0)}"]
        for r, (j, name) in enumerate(((0, "level (mean IHS4)"), (1, "oscillation (mean |change| between visits)"))):
            ax = axes[r, c]
            x, y = np.maximum(D[:, j], 0.3), np.maximum(G[:, j], 0.3)
            lim = (0.25, max(x.max(), y.max()) * 1.3)
            ax.plot(lim, lim, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.scatter(x, y, s=26, facecolor=SERIES[0], edgecolor=SURFACE, linewidth=0.8, alpha=0.85, zorder=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.grid(color=GRID, linewidth=0.6, which="both")
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xlabel(f"data: {name}")
            ax.set_ylabel(f"generated: {name}")
            ax.set_title(f"{title}\n{name.split(' (')[0]}", loc="left", fontsize=10)
            ax.text(0.03, 0.97, f"median: generated {np.median(G[:, j]):.2f}, data {np.median(D[:, j]):.2f}\nSpearman {spearmanr(D[:, j], G[:, j])[0]:.2f}"
                    f"; 90th pct: generated {np.percentile(G[:, j], 90):.1f}, data {np.percentile(D[:, j], 90):.1f}",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK2, linespacing=1.4)
    fig.suptitle("Generation check: 100 training patients, 36 months from the last visit in the first 12 months (scheduled visits); "
                 "no diffusion of z,\ndrawn flares, visit noise; one dot per patient (mean of 5 generated courses); dashed line: generated = data",
                 x=0.06, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f"{run}/generation_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
