"""
Figure: the stage 1 flare model (flare_model.py) on the 8 code archetypes of plot_code_archetypes.py (same patients, same colours; the
counterpart of plot_code_rollouts.py). Per patient: the visit scores and the flare onsets (flare visits, ticks), the fitted slow baseline
b(t), the fit b(t) + F(t) with the observed flares, and n generated courses: the fitted baseline with flares drawn from the fitted process
(rate, amplitude and duration from the patient code and the own drug path), no visit noise. The slow baseline is fitted, not generated
(stage 2: the SDE). Printed: level and oscillation at the scheduled visits (background, treatment monitoring), data against the
generated courses with visit noise. CPU, about 1 min.

    python -m experiments.latent.plot_flare_model [runs/flare_model] [runs/hs_fused_sde_<stamp>] [n]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.latent import flare_model as fm  # noqa: E402
from experiments.latent.hs_e2e import MONTH  # noqa: E402
from experiments.latent.plot_code_archetypes import GRID, INK, INK2, MUTED, SERIES, SURFACE, archetypes, load  # noqa: E402


def draw_flares(fmod, th, c, rna, cells, rng):
    """onsets, log-amplitudes and durations drawn from the fitted process on the monthly cells (t mid, dt, drugs) of one patient"""
    t, dt, u = cells
    bio = (u.sum(1) > 0).astype(float)
    nu = np.exp(float(fmod.a0) + float(fmod.g_t) * t / fm.YEAR + c[0] + u @ th["b_nu"] + bio * rna[0])
    k = rng.poisson(nu * dt / fm.YEAR)
    tau = np.repeat(t - dt / 2, k) + rng.random(k.sum()) * np.repeat(dt, k)
    vm = fmod.v_mean.numpy()  # the centring of the model; vm[3]: the share of onsets with an antibiotic (data)
    V = np.column_stack([np.repeat(u, k, 0), (rng.random(k.sum()) < vm[3]).astype(float)]) - vm
    b = np.repeat(bio, k)
    logA = float(fmod.mu_A) + c[1] + V @ th["b_A"] + b * rna[1] + float(fmod.ls_A.exp()) * rng.standard_normal(k.sum())
    D = np.exp(float(fmod.d0) + c[2] + V @ th["b_D"] + b * rna[2])
    return tau, logA, D


def excess(t, tau, logA, D):
    lag = t[:, None] - tau[None, :]
    return np.where((lag >= 0) & (lag <= fm.REACH), np.exp(logA[None, :] - np.clip(lag, 0, None) / D[None, :]), 0.0).sum(1)


def main():
    fdir = sys.argv[1] if len(sys.argv) > 1 else "runs/flare_model"
    run = sys.argv[2] if len(sys.argv) > 2 else "runs/hs_fused_sde_20261026-234554"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    torch.set_num_threads(2)
    ck, data, model = load(run)
    tr, groups, *_ = archetypes(ck, data, model)
    short = {"High level": "High", "Low level": "Low", "Stable (middle level)": "Stable", "Oscillatory (middle level)": "Oscill."}
    sel = [(f"{short[g]} {'AB'[j]}", str(data.pid[int(tr[k])])) for g, ks in groups.items() for j, k in enumerate(ks)]
    del ck, data, model
    d = fm.load()
    fck = torch.load(f"{fdir}/model.pt", weights_only=False)
    fmod = fm.FlareModel(d, rw=fck["rw"], nc=fck.get("nc", False))
    fmod.load_state_dict(fck["model"])
    with torch.no_grad():
        r = fmod.parts(d, sample=False)
        th = {k: v.numpy() for k, v in r["th"].items()}
        c, logD, rna = r["c"].numpy(), r["logD"].numpy(), (d["R"] @ r["th"]["g_R"]).numpy()
        a_m = r["logA"].numpy()
    pidx = {p: i for i, p in enumerate(d["pats"])}
    k_pat = d["k_p"].numpy()
    np_ = {k: np.asarray(d[k]) for k in ("o_p", "o_t", "o_y", "o_sched", "f_p", "f_t", "c_p", "c_t", "c_dt", "c_u")}
    km = fmod.k_m.detach().numpy()
    rng = np.random.default_rng(0)
    sig_y = float(fmod.ls_y.exp())

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharey=True)
    ymax = max(np.expm1(np_["o_y"][np_["o_p"] == pidx[p]]).max() for _, p in sel) * 1.1
    print(f"flare model {fdir}; {n} generated courses each; level / oscillation (mean |change|) at the scheduled visits")
    print(f"{'patient':10s} {'id':>10s} {'visits':>6s} {'data':>13s} {'generated':>13s} {'flares/yr data':>15s} {'model':>6s}")
    for q, (ax, (name, p)) in enumerate(zip(axes.T.ravel(), sel)):
        i, col = pidx[p], SERIES[q]
        kk = np.flatnonzero(k_pat == i)
        cm = np_["c_p"] == i
        cells = (np_["c_t"][cm], np_["c_dt"][cm], np_["c_u"][cm])
        span = cells[0][-1] + cells[1][-1] / 2
        t = np.arange(0.0, span + 1, 3.0)
        cell_u = cells[2][np.minimum((t // fm.CELL).astype(int), len(cells[0]) - 1)]
        b = np.interp(t, np.arange(len(kk)) * fm.KNOT, km[kk]) + cell_u @ th["b_L"]
        fk = np.flatnonzero(np_["f_p"] == i)
        F = excess(t, np_["f_t"][fk], a_m[fk], np.exp(logD[fk]))
        om = np_["o_p"] == i
        ot, oy, osch = np_["o_t"][om], np.expm1(np_["o_y"][om]), np_["o_sched"][om]
        b_o = np.interp(ot, t, b)
        gen_o = []
        for s in range(n):
            tau, la, D = draw_flares(fmod, th, c[i], rna[i], cells, rng)
            ax.plot(t / MONTH, np.expm1(b + excess(t, tau, la, D)), color=col, linewidth=0.9, alpha=0.4, zorder=1)
            gen_o.append(np.expm1(b_o + excess(ot, tau, la, D) + rng.laplace(0, sig_y, len(ot))).clip(min=0))
        ax.plot(t / MONTH, np.expm1(b), color=col, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2)
        ax.plot(t / MONTH, np.expm1(b + F), color=col, linewidth=1.6, zorder=3)
        ax.scatter(ot / MONTH, oy, s=30, facecolor=col, edgecolor=SURFACE, linewidth=1.0, zorder=4)
        ax.plot(np_["f_t"][fk] / MONTH, np.zeros(len(fk)), linestyle="", marker="|", markersize=9, color=INK2, clip_on=False, zorder=5)
        ax.set_title(f"{name}  ({p})", loc="left", fontsize=10)
        ax.set_ylim(0, ymax)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if q % 2 == 1:
            ax.set_xlabel("months since the first visit")
        if q < 2:
            ax.set_ylabel("IHS4")
        g = np.array(gen_o)[:, osch]
        lo = lambda y: (y.mean(-1), np.abs(np.diff(y, axis=-1)).mean(-1)) if y.shape[-1] > 1 else (y.mean(-1), np.nan)
        (dl, do), (gl, go) = lo(oy[osch]), lo(g)
        print(f"{name:10s} {p:>10s} {osch.sum():6d} {dl:6.1f} / {do:4.1f} {np.mean(gl):6.1f} / {np.nanmean(go):4.1f} "
              f"{len(fk) / (span / fm.YEAR):15.2f} {np.sum(np.exp(float(fmod.a0) + float(fmod.g_t) * cells[0] / fm.YEAR + c[i, 0] + cells[2] @ th['b_nu'] + (cells[2].sum(1) > 0) * rna[i, 0]) * cells[1]) / span:6.2f}")
    key = [plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=5, label="visit score (data)"),
           plt.Line2D([], [], color=INK2, marker="|", linestyle="", markersize=9, label="flare visit (onset, data)"),
           plt.Line2D([], [], color=INK2, linewidth=1.6, label="fit: baseline + observed flares"),
           plt.Line2D([], [], color=INK2, linewidth=1.6, linestyle=(0, (4, 2)), label="fitted slow baseline"),
           plt.Line2D([], [], color=INK2, linewidth=0.9, alpha=0.5, label=f"generated: baseline + drawn flares ({n})")]
    fig.legend(handles=key, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Flare model on the 8 code archetypes: slow baseline + flares (onset at the flare visit, amplitude, exponential decay)",
                 x=0.06, ha="left", fontsize=11.5, color=INK)
    fig.text(0.06, 0.935, "generated courses: flares drawn from the patient's fitted rate, amplitude and duration on the own drug path; the "
             "slow baseline is the fitted one (stage 2: the SDE); no visit noise in the lines", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = f"{fdir}/flare_archetypes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
