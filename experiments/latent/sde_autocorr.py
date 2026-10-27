"""
Does the latent SDE learn the time correlation of the scores, and does it change with treatment? (GPU, < 1 min)
For held-out patients: one start per patient (a random visit with >= 3 scored visits in the next 12 months and one constant plan over them). The SDE
gives the expected visit score m(t) at each later visit day (mean of 56 trajectories). Residual r = score - m(t), for the observed visits and for 8
further generated trajectories at the same days. Autocorrelation rho(lag) = corr(r_i, r_j) over the visit pairs of a start, in bins of the lag.
If the SDE has the right dynamics, the generated rho(lag) follows the observed one; per plan group: no biologic, Adalimumab, Bimekizumab / Secukinumab.
Writes <out>/D_autocorrelation.png.

    python -m experiments.latent.sde_autocorr runs/hs_e2e_<stamp> --sde runs/sde_v2.pt --out runs/hs_e2e_<stamp>/sde_plots_v2
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.latent.hs_e2e import MONTH, VISIT
from experiments.latent.hs_task2 import load
from experiments.latent.sde_fit import LatentSDE, simulate
from experiments.latent.sde_plots import C_SDE, C_TRUTH, style
from experiments.latent.sde_states import plan_of_span

BINS = [(0.2, 1), (1, 2), (2, 4), (4, 6), (6, 9), (9, 12)]
GROUPS = {"no biologic": (0,), "Adalimumab": (1,), "Bimekizumab / Secukinumab": (2, 3)}
N_MEAN, N_GEN = 56, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sde", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, _ = load(args.run, device)
    Z = torch.as_tensor(np.load(f"{args.run}/sde_cache.npz")["Z"], device=device)
    sde = LatentSDE().to(device)
    sde.load_state_dict(torch.load(args.sde or f"{args.run}/sde.pt", map_location=device))
    sde.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    rng = np.random.default_rng(0)
    starts = []
    for i in np.flatnonzero(data.flag):
        s0, s1 = data.off[i], data.off[i + 1]
        vr = s0 + np.flatnonzero((data.kind[s0:s1] == VISIT) & data.has_score[s0:s1])
        days = data.day[vr]
        st, en, dr = data.bio[i]
        ok = [r for r, t in zip(vr, days) if t <= data.end_day - 365 and ((days > t) & (days <= t + 365)).sum() >= 3]
        rng.shuffle(ok)
        for r in ok[:5]:
            t0 = data.day[r]
            p = plan_of_span(st, en, dr, t0, t0 + 365)
            if p >= 0:
                fut = (days > t0) & (days <= t0 + 365)
                starts.append((r, p, (days[fut] - t0)[:12], data.score[vr][fut][:12]))
                break
    print(f"{len(starts)} held-out starts; plans: " + ", ".join(f"{g} {sum(s[1] in ps for s in starts)}" for g, ps in GROUPS.items()))
    obs_r, gen_r, lags, grp = [], [], [], []
    with torch.no_grad():
        for a in range(0, len(starts), 300):
            part = starts[a:a + 300]
            days, valid = np.zeros((len(part), 12), np.int64), np.zeros((len(part), 12), bool)
            for k, s in enumerate(part):
                days[k, :len(s[2])], valid[k, :len(s[2])] = s[2], True
            sc = simulate(sde, Z[[s[0] for s in part]], torch.as_tensor([s[1] for s in part], device=device), torch.as_tensor(days, device=device),
                          N_MEAN + N_GEN, gen=gen).cpu().numpy()
            m = sc[:, N_GEN:].mean(1)  # expected visit score, from trajectories not used as pseudo-observations
            for k, s in enumerate(part):
                n = len(s[2])
                ro, rg = s[3] - m[k, :n], sc[k, :N_GEN, :n] - m[k, :n]
                for i_ in range(n):
                    for j_ in range(i_ + 1, n):
                        obs_r.append((ro[i_], ro[j_])); gen_r.append(np.stack([rg[:, i_], rg[:, j_]], 1)); lags.append((s[2][j_] - s[2][i_]) / MONTH)
                        grp.append(s[1])
    obs_r, gen_r, lags, grp = np.array(obs_r), np.array(gen_r), np.array(lags), np.array(grp)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (g, ps) in zip(axes, GROUPS.items()):
        print(f"\n{g}: residual sd observed {np.std(obs_r[np.isin(grp, ps), 0]):.2f}, generated {np.std(gen_r[np.isin(grp, ps), :, 0]):.2f}")
        print(f"   {'lag (months)':>13s} {'pairs':>6s} {'observed rho':>13s} {'generated rho':>14s}")
        xs, ro_, rlo, rhi, rg_ = [], [], [], [], []
        for lo, hi in BINS:
            s = np.isin(grp, ps) & (lags >= lo) & (lags < hi)
            if s.sum() < 30:
                continue
            o = obs_r[s]
            r_obs = np.corrcoef(o[:, 0], o[:, 1])[0, 1]
            bs = [np.corrcoef(*o[rng.integers(0, len(o), len(o))].T)[0, 1] for _ in range(300)]
            r_gen = np.mean([np.corrcoef(gen_r[s, t, 0], gen_r[s, t, 1])[0, 1] for t in range(N_GEN)])
            print(f"   {f'{lo}-{hi}':>13s} {s.sum():6d} {r_obs:13.3f} {r_gen:14.3f}")
            xs.append((lo + hi) / 2); ro_.append(r_obs); rlo.append(np.percentile(bs, 2.5)); rhi.append(np.percentile(bs, 97.5)); rg_.append(r_gen)
        ax.fill_between(xs, rlo, rhi, color="#d4d4d4", lw=0, label="observed, 95% interval (pairs resampled)")
        ax.plot(xs, ro_, "-o", color=C_TRUTH, lw=2, ms=5, label="observed")
        ax.plot(xs, rg_, "-o", color=C_SDE, lw=2, ms=5, label="generated (SDE)")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(f"{g} ({np.isin(grp, ps).sum()} visit pairs)", loc="left", fontsize=10)
        ax.set_xlabel("months between the two visits")
        style(ax)
    axes[0].set_ylabel("correlation of the residuals")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Time correlation of the scores around the expected course (held-out patients)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = args.out or f"{args.run}/sde_plots"
    fig.savefig(f"{out}/D_autocorrelation.png", dpi=130)
    print(f"\nplot: {out}/D_autocorrelation.png")


if __name__ == "__main__":
    main()
