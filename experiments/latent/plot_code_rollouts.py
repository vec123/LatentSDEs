"""
Figure: SDE rollouts of the 8 code archetypes of plot_code_archetypes.py (patient-code model; no training; CPU, seconds).

Each patient: the SDE prior from the posterior at the first step to the last record day (7-day grid), with the patient's trained code and
the patient's own treatment path (the treatment flags of the last record step at or before each grid point). n paths with diffusion
(start state sampled from the posterior, Euler-Maruyama noise) and n paths without diffusion (start state sampled, drift only): the
difference of the two bundles is the diffusion alone. Shown: the latent severity g(z_t) (no visit-type effects) against the visit scores.
The codes of these training patients are trained on the whole record: descriptive, not a forecast.
Printed: per patient, the level (mean at the scored visits) and the oscillation (mean |change| between successive scored visits) of the
data and of the paths.

Flare models (--flare 1): + the flare excess (observed flares at the start, new onsets drawn from the rate of the state; drift-only paths:
the expected flare burden).

    python -m experiments.latent.plot_code_rollouts [runs/hs_fused_sde_<stamp>] [n_paths] [selection run]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.latent.hs_e2e import MONTH  # noqa: E402
from experiments.latent.hs_fused_sde import STEP_DAYS, extras, flare_terms  # noqa: E402
from experiments.latent.plot_code_archetypes import GRID, INK, INK2, MUTED, SERIES, SURFACE, archetypes, load  # noqa: E402


def rollouts(model, data, i, m0, lv0, n, noise, gen, F0=0.0, zdiff=True, months=None, k0=0, dscale=1.0):
    """n paths of g(z_t) for patient i on the 7-day grid from N(m0, exp(lv0)); returns the grid (days) and (n, grid) scores. Flare models:
    + the flare excess, from F0 (the observed flares at the start) and new onsets drawn from the rate of the state with prior amplitudes
    (patient flare code, treatment of the grid point); noise False: the expected flare burden instead; zdiff False: z without diffusion
    (drift only), the flares still drawn; k0: the start row (local index; m0, lv0, F0 belong to it); dscale: a factor on the stationary
    variance of z (the noise of each step x sqrt(dscale); the calibration of the diffusion)"""
    s_ = slice(data.off[i], data.off[i + 1])
    day = data.day[s_]
    grid = np.arange(day[k0], (min(day[-1], day[k0] + months * MONTH) if months else day[-1]) + STEP_DAYS, STEP_DAYS)
    rows = data.off[i] + np.searchsorted(day, grid, side="right") - 1  # the last record step at or before each grid point
    u = torch.as_tensor(data.u[rows], dtype=torch.float32)
    s = torch.as_tensor(data.s[i], dtype=torch.float32).expand(n, -1)
    c = model.pcode.weight[i].detach().expand(n, -1) if model.d_code else None
    lm, ls = data.stats["lm"], data.stats["ls"]
    z = m0 + (0.5 * lv0).exp() * torch.randn((n, model.dz), generator=gen)
    F = torch.full((n,), float(F0))
    out = [torch.expm1(model.decode(z)[0] * ls + lm + F).clamp(min=0.0)]
    if model.flare:
        sc = model.fl_ls_c.exp() * model.pflare.weight[i].detach()
        sA = model.fl_ls_A.exp()
    for g in range(1, len(grid)):
        ug = u[g - 1].expand(n, -1)
        zm, _, add = model.step(z, torch.zeros_like(z), ug, s, STEP_DAYS / MONTH, c)
        z = (zm + ((dscale * add).sqrt() * torch.randn(z.shape, generator=gen) if noise and zdiff else 0.0)).clamp(-20.0, 20.0)
        if model.flare:
            a_in = torch.cat([ug[:, [0, 2, 3, 4]], s], -1)
            pA = model.fl_amp_p(a_in)[:, 0] + sc[1]
            decay = torch.exp(-STEP_DAYS / (model.fl_dur(a_in)[:, 0] + sc[2]).exp())
            nu = torch.exp((model.fl_rate(torch.cat([z, ug[:, 2:5], s], -1))[:, 0] + sc[0]).clamp(max=6.0)) * STEP_DAYS / 365.25
            if noise:
                on = (torch.rand(n, generator=gen) < 1 - torch.exp(-nu)).float()
                F = F * decay + on * torch.exp((pA + sA * torch.randn(n, generator=gen)).clamp(-8.0, 2.0))
            else:
                F = F * decay + (1 - torch.exp(-nu)) * torch.exp(pA + 0.5 * sA ** 2)  # at most one onset per step, as when drawn
        out.append(torch.expm1(model.decode(z)[0] * ls + lm + F).clamp(min=0.0))
    return grid, torch.stack(out, 1).numpy()


def level_osc(y):
    return y.mean(-1), np.abs(np.diff(y, axis=-1)).mean(-1)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_fused_sde_20261026-234554"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sel_run = sys.argv[3] if len(sys.argv) > 3 else run  # the model whose codes select the 8 patients (the same patients across models)
    nodiff = len(sys.argv) > 4 and sys.argv[4] == "1"  # the thin paths without the diffusion of z (flares still drawn)
    start = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0  # start at the last step in the first `start` months (0: the first visit)
    s0 = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0  # a factor on the posterior sd of the start state (0: the posterior mean)
    horizon = 36 if start else None
    torch.set_num_threads(2)
    tr, groups, *_ = archetypes(*load(sel_run))
    ck, data, model = load(run)
    short = {"High level": "High", "Low level": "Low", "Stable (middle level)": "Stable", "Oscillatory (middle level)": "Oscill."}
    sel = [(f"{short[g]} {'AB'[j]}", int(tr[k])) for g, ks in groups.items() for j, k in enumerate(ks)]
    gen = torch.Generator().manual_seed(0)
    res = {}
    with torch.no_grad():
        pats = [i for _, i in sel]
        model.encode(data.pack(pats, "cpu"))
        m, lv = model.last_post
        Fp = np.zeros(m.shape[:2], np.float32)
        if model.flare:  # the excess of the observed flares at the start (mean amplitudes, the trained flare code)
            Fp = flare_terms(model, extras(data, pats, "cpu"), model.pflare.weight[pats].detach(), sample=False)["F_post"].numpy()
        for b, (name, i) in enumerate(sel):
            d = data.day[data.off[i]:data.off[i + 1]]
            k0 = int(np.searchsorted(d - d[0], start * MONTH, side="right")) - 1
            lvb = lv[b, k0] + (2 * np.log(s0) if s0 > 0 else -60.0)
            grid, wn = rollouts(model, data, i, m[b, k0], lvb, n, True, gen, Fp[b, k0], zdiff=not nodiff, months=horizon, k0=k0)
            _, nn_ = rollouts(model, data, i, m[b, k0], lvb, n, False, gen, Fp[b, k0], months=horizon, k0=k0)
            res[name] = (i, grid, wn, nn_)

    print(f"model {run}; {n} paths each; level = mean at the scored visits, oscillation = mean |change| between successive scored visits")
    lab = (("with diffusion and drawn flares", "drift + expected flare burden") if model.flare else
           ("with diffusion", "without diffusion, drift only"))
    if nodiff:
        lab = ("without diffusion, drift + drawn flares", lab[1])
    print(f"{'patient':10s} {'id':>10s} {'data level/osc':>15s} {lab[0][:15]:>15s} {lab[1][:15]:>15s}")
    for name, (i, grid, wn, nn_) in res.items():
        s_ = slice(data.off[i], data.off[i + 1])
        h = data.has_score[s_] & (data.day[s_] > grid[0]) & (data.day[s_] <= grid[-1])  # the scored visits in the generated window
        if h.sum() < 2:
            continue
        at = np.clip(np.searchsorted(grid, data.day[s_][h]), 0, len(grid) - 1)  # the grid point at or after each scored visit
        dl, do = level_osc(data.score[s_][h])
        (wl, wo), (nl, no) = level_osc(wn[:, at]), level_osc(nn_[:, at])
        print(f"{name:10s} {data.pid[i]:>10s} {dl:6.1f} / {do:5.1f}  {wl.mean():6.1f} / {wo.mean():5.1f}  {nl.mean():6.1f} / {no.mean():5.1f}")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharey=True)
    sc_max = max(data.score[data.off[i]:data.off[i + 1]][data.has_score[data.off[i]:data.off[i + 1]]].max() for i, *_ in res.values())
    ymax = max(sc_max, max(np.percentile(np.concatenate([wn, nn_]), 98) for _, _, wn, nn_ in res.values())) * 1.05
    for q, (ax, (name, (i, grid, wn, nn_))) in enumerate(zip(axes.T.ravel(), res.items())):
        col = SERIES[q]
        s_ = slice(data.off[i], data.off[i + 1])
        h = data.has_score[s_]
        t0 = data.day[s_][0]
        tg = (grid - t0) / MONTH
        for p in wn:
            ax.plot(tg, p, color=col, linewidth=0.9, alpha=0.4, zorder=1)
        for p in nn_:
            ax.plot(tg, p, color=col, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2)
        ax.scatter((data.day[s_][h] - t0) / MONTH, data.score[s_][h], s=30, facecolor=col, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        if start:  # the window: the data up to the end of the generation, a line at the start
            ax.axvline(tg[0], color=INK2, linewidth=0.8, linestyle=":", zorder=0)
            ax.set_xlim(-1, tg[-1] + 1)
        ax.set_title(f"{name}  ({data.pid[i]})", loc="left", fontsize=10)
        ax.set_ylim(0, ymax)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if q % 2 == 1:
            ax.set_xlabel("months since the first visit")
        if q < 2:
            ax.set_ylabel("IHS4")
    key = [plt.Line2D([], [], color=INK2, marker="o", linestyle="", markersize=5, label="visit score (data)"),
           plt.Line2D([], [], color=INK2, linewidth=0.9, alpha=0.5, label=f"SDE rollout {lab[0]} ({n} paths)"),
           plt.Line2D([], [], color=INK2, linewidth=1.6, linestyle=(0, (4, 2)), label=f"rollout {lab[1]} ({n} paths)")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
    where = f"from the last visit in the first {start:g} months (dotted line), 36 months" if start else "from the first visit"
    how = "start state = the posterior mean" if s0 == 0 else ("start state drawn from the posterior" + (f" (sd x {s0:g})" if s0 != 1 else ""))
    fig.suptitle(f"SDE rollouts of the 8 code archetypes {where}: patient code and own treatment path; latent severity g(z_t) "
                 + ("+ flare excess " if model.flare else "") + "without visit-type effects", x=0.06, ha="left", fontsize=11.5, color=INK)
    fig.text(0.06, 0.935, f"{how}; no visit noise in the lines; training patients (codes trained on the whole record: descriptive); the y "
             "axis is cut at the 98th percentile of the paths", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = f"{run}/code_rollouts{'_nodiff' if nodiff else ''}{f'_start{start:g}' if start else ''}{'_mean' if s0 == 0 else ''}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
