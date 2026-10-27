"""
Figure: very different patient codes and their trajectories (patient-code model; no training; CPU, seconds).

Training patients with >= 15 scored visits over >= 24 months. Selection by the code only: a ridge from the standardized code to the patient's
mean visit score (level) and to the mean absolute change between successive scored visits (oscillation: flares up and down; a trend or one
large drop counts little); then the 2 highest and 2 lowest predicted levels, and among patients with a middle predicted level (40-60th
percentile) the 2 lowest and 2 highest predicted oscillations. Shown: the visit scores (data, dots) and the
latent severity g(z_t) of the model (posterior mean decoded without visit-type effects, line); the overview places them in the code space.

    python -m experiments.latent.plot_code_archetypes [runs/hs_fused_sde_<stamp>]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.latent.hs_e2e import MONTH, PATHS, Data  # noqa: E402
from experiments.latent.hs_fused_sde import make_model, prepare  # noqa: E402

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#b4b3ad"
# one colour per patient, the same in the code space and in the time series: the 8 categorical slots of the reference palette in fixed order
# (validated; the two patients of a panel are adjacent slots; aqua, yellow, magenta are below 3:1 contrast, so every mark is labelled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def load(run):
    ck = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"])
    model.load_state_dict(ck["model"])
    model.eval()
    return ck, data, model


def archetypes(ck, data, model):
    """the selection of the figure (module docstring): the training patients, the data level and oscillation, their ridge predictions
    from the code, and the 8 archetypes {group: 2 indices into the training patients}"""
    pool = np.random.default_rng(ck["args"].get("seed", 0)).permutation(np.flatnonzero(~data.flag))
    tr = np.sort(pool[int(0.1 * len(pool)):])
    C = model.pcode.weight.detach().numpy()[tr]

    lvl, sd, nsc, span = (np.full(len(tr), np.nan) for _ in range(4))
    for k, i in enumerate(tr):
        s = slice(data.off[i], data.off[i + 1])
        h = data.has_score[s]
        sc, dy = data.score[s][h], data.day[s][h]
        nsc[k] = len(sc)
        if len(sc):
            lvl[k], span[k] = sc.mean(), (dy.max() - dy.min()) / MONTH
        if len(sc) > 2:
            sd[k] = np.abs(np.diff(sc)).mean()  # oscillation: mean absolute change between successive scored visits
    Z = StandardScaler().fit_transform(C)
    ok = np.isfinite(sd) & np.isfinite(lvl)
    lvl_hat = Ridge(1.0).fit(Z[ok], lvl[ok]).predict(Z)
    sd_hat = Ridge(1.0).fit(Z[ok], sd[ok]).predict(Z)
    el = np.flatnonzero(ok & (nsc >= 15) & (span >= 24))
    mid = el[(lvl_hat[el] >= np.percentile(lvl_hat[el], 40)) & (lvl_hat[el] <= np.percentile(lvl_hat[el], 60))]
    groups = {"High level": el[np.argsort(-lvl_hat[el])[:2]], "Low level": el[np.argsort(lvl_hat[el])[:2]],
              "Stable (middle level)": mid[np.argsort(sd_hat[mid])[:2]], "Oscillatory (middle level)": mid[np.argsort(-sd_hat[mid])[:2]]}
    return tr, groups, lvl, sd, nsc, lvl_hat, sd_hat, ok


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_fused_sde_20261026-234554"
    torch.set_num_threads(2)
    ck, data, model = load(run)
    tr, groups, lvl, sd, nsc, lvl_hat, sd_hat, ok = archetypes(ck, data, model)
    r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"in-sample R2 from the code: level {r2(lvl[ok], lvl_hat[ok]):.2f}, oscillation {r2(sd[ok], sd_hat[ok]):.2f}")

    sel = [int(tr[k]) for g in groups.values() for k in g]
    with torch.no_grad():
        model.encode(data.pack(sel, "cpu"))
        mu = model.decode(model.last_post[0])[0].numpy()
    G = {i: np.expm1(mu[b, :data.n_rows(i)] * data.stats["ls"] + data.stats["lm"]) for b, i in enumerate(sel)}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig = plt.figure(figsize=(13.5, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.34, hspace=0.75)
    # the code projected on its level and oscillation directions (the two ridge maps of the selection); PC1 x PC2 hides the oscillation,
    # which lies in smaller directions of the 16-d code
    P = np.column_stack([lvl_hat, sd_hat])
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.scatter(P[:, 0], P[:, 1], s=3, color=MUTED, alpha=0.35, linewidths=0, rasterized=True)
    ax0.set_title("Code space: level and oscillation directions\n(all training codes)", loc="left", fontsize=10)
    ax0.set_xlabel(f"level predicted from the code (IHS4; R2 {r2(lvl[ok], lvl_hat[ok]):.2f})")
    ax0.set_ylabel(f"oscillation predicted from the code\n(mean |change| between visits; R2 {r2(sd[ok], sd_hat[ok]):.2f})")
    ax0.grid(color=GRID, linewidth=0.6)
    ax0.set_axisbelow(True)
    short = {"High level": "High", "Low level": "Low", "Stable (middle level)": "Stable", "Oscillatory (middle level)": "Oscill."}
    for q, (g, ks) in enumerate(groups.items()):
        for j, k in enumerate(ks):
            ax0.scatter(P[k, 0], P[k, 1], s=70, facecolor=SERIES[2 * q + j], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax0.annotate(f"{short[g]} {'AB'[j]}", (P[k, 0], P[k, 1]), xytext=(-7, 3) if j == 0 else (7, -3), textcoords="offset points",
                         ha="right" if j == 0 else "left", va="bottom" if j == 0 else "top", color=INK, fontsize=8.5, zorder=4)

    ymax = max(max(G[i].max() for i in sel), max(data.score[data.off[i]:data.off[i + 1]][data.has_score[data.off[i]:data.off[i + 1]]].max()
                                                  for i in sel)) * 1.05
    for q, ((g, ks), pos) in enumerate(zip(groups.items(), [(0, 1), (0, 2), (1, 1), (1, 2)])):
        ax = fig.add_subplot(gs[pos])
        for j, k in enumerate(ks):
            i = int(tr[k])
            s = slice(data.off[i], data.off[i + 1])
            t = (data.day[s] - data.day[s][0]) / MONTH
            h = data.has_score[s]
            col = SERIES[2 * q + j]
            ax.plot(t, G[i], color=col, linewidth=1.6, zorder=2)
            ax.scatter(t[h], data.score[s][h], s=22, facecolor=col, edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
            ax.annotate(f"{short[g]} {'AB'[j]}", (t[-1], G[i][-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=8, annotation_clip=False)
        ax.set_title(f"{g}", loc="left", fontsize=10, pad=38)
        ax.text(0, 1.02, f"A {data.pid[int(tr[ks[0]])]}, B {data.pid[int(tr[ks[1]])]}\npredicted from the code: level {lvl_hat[ks].mean():.1f}, "
                f"oscillation {sd_hat[ks].mean():.1f}\ndata: level {lvl[ks].mean():.1f}, oscillation {sd[ks].mean():.1f}",
                transform=ax.transAxes, va="bottom", fontsize=7.5, color=INK2, linespacing=1.3)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("months since the first visit")
        ax.set_ylabel("IHS4")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for sp in ("top", "right"):
        ax0.spines[sp].set_visible(False)
    key = [plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=5, label="visit score (data)"),
           plt.Line2D([], [], color=INK2, linewidth=1.6, label="latent severity g(z_t), no visit-type effect (model)")]
    fig.legend(handles=key, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.62, -0.005))
    fig.suptitle("Patients with very different codes, and their courses (selected by the code alone; training patients)", x=0.07,
                 ha="left", fontsize=12, color=INK)
    out = f"{run}/code_archetypes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    for g, ks in groups.items():
        print(f"{g:28s} " + ", ".join(f"{data.pid[int(tr[k])]} (data level {lvl[k]:.1f}, oscillation {sd[k]:.1f}, {int(nsc[k])} scores)"
                                     for k in ks))


if __name__ == "__main__":
    main()
