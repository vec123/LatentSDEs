"""
Generation from a better-known state and the calibration of the diffusion of z by one factor (no training; CPU, ~2 min).

100 random training patients with >= 4 scored scheduled visits (background, treatment monitoring) in the 36 months after the start; the
start is the last record step in the first 12 months (the posterior there has seen a year of data). From the start state (drawn from the
posterior there; flare models: + the excess of the observed flares) 36 months of generation on the 7-day grid with the own treatment path:
  mean path   from the posterior mean, no noise, the expected flare burden: error at the scheduled visits by months after the start
  k = ...     5 courses per patient with drawn flares and the diffusion of z x sqrt(k) (k = 0: drift only), + visit noise Laplace 0.155 in
              log1p (the near-replicate value): per patient the level (mean) and the oscillation (mean |change| between successive
              scheduled visits) against the data; medians, Spearman over the patients, 90th percentile of the level
The calibrated k: the median oscillation of the generated courses = the data (interpolated in log k between the grid values).

s0 (3rd argument, default 1): a factor on the posterior sd of the start state (0: the posterior mean; a narrower start).

    python -m experiments.latent.diffusion_calibration runs/hs_fused_sde_<stamp> [k1,k2,...] [s0]
"""
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.latent.hs_e2e import MONTH
from experiments.latent.hs_fused_sde import extras, flare_terms
from experiments.latent.plot_code_archetypes import load
from experiments.latent.plot_code_rollouts import rollouts

START, HORIZON, NOISE = 12, 36, 0.155


def main():
    run = sys.argv[1]
    ks = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.0, 0.03, 0.1, 0.3, 1.0]
    s0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    lv_shift = 2 * np.log(s0) if s0 > 0 else -60.0  # log variance of the start state + this
    torch.set_num_threads(2)
    ck, data, model = load(run)
    pool = np.random.default_rng(ck["args"].get("seed", 0)).permutation(np.flatnonzero(~data.flag))
    tr = np.sort(pool[int(0.1 * len(pool)):])
    ok = []
    for i in tr:
        s_ = slice(data.off[i], data.off[i + 1])
        d = data.day[s_] - data.day[s_][0]
        k0 = int(np.searchsorted(d, START * MONTH, side="right")) - 1
        win = (d > d[k0]) & (d <= d[k0] + HORIZON * MONTH) & data.has_score[s_] & np.isin(data.vt[s_], (0, 3))
        if k0 >= 1 and win.sum() >= 4:
            ok.append((int(i), k0))
    sel = [ok[j] for j in np.sort(np.random.default_rng(1).choice(len(ok), min(100, len(ok)), replace=False))]
    gen = torch.Generator().manual_seed(0)
    res = {k: [] for k in ks}
    dat, bias = [], []
    with torch.no_grad():
        for a in range(0, len(sel), 50):
            chunk = sel[a:a + 50]
            pats = [i for i, _ in chunk]
            model.encode(data.pack(pats, "cpu"))
            m, lv = model.last_post
            Fp = (flare_terms(model, extras(data, pats, "cpu"), model.pflare.weight[pats].detach(), sample=False)["F_post"].numpy()
                  if model.flare else np.zeros((len(pats), m.shape[1])))
            for b, (i, k0) in enumerate(chunk):
                s_ = slice(data.off[i], data.off[i + 1])
                d = data.day[s_]
                js = np.flatnonzero((d > d[k0]) & (d <= d[k0] + HORIZON * MONTH) & data.has_score[s_] & np.isin(data.vt[s_], (0, 3)))
                t, y = d[js], data.score[s_][js]
                dat.append([y.mean(), np.abs(np.diff(y)).mean()])
                gr, p = rollouts(model, data, i, m[b, k0], torch.full_like(m[b, k0], -30.0), 1, False, gen, Fp[b, k0], months=HORIZON, k0=k0)
                at = np.clip(np.searchsorted(gr, t), 0, len(gr) - 1)
                bias += [((t[j] - d[k0]) / MONTH, p[0, at[j]] - y[j]) for j in range(len(js))]
                for k in ks:
                    gr, q = rollouts(model, data, i, m[b, k0], lv[b, k0] + lv_shift, 5, True, gen, Fp[b, k0], zdiff=k > 0, months=HORIZON,
                                     k0=k0, dscale=k)
                    g = np.expm1(np.log1p(q[:, at]) + np.random.default_rng(i).laplace(0, NOISE, (5, len(js)))).clip(min=0)
                    res[k].append([g.mean(1).mean(), np.abs(np.diff(g, axis=1)).mean(1).mean()])
    D = np.array(dat)
    bias = np.array(bias)
    print(f"{run}: {len(sel)} training patients; start = the last step in the first {START} months (posterior sd x {s0}); {HORIZON} "
          f"months of generation")
    print("mean path (posterior mean at the start, no noise, expected flare burden): error at the scheduled visits (rollout - data)")
    for lo, hi in ((0, 6), (6, 12), (12, 24), (24, 36.01)):
        s = (bias[:, 0] > lo) & (bias[:, 0] <= hi)
        print(f"   months {lo:2d}-{hi:<5.0f} n={s.sum():5d}  mean {bias[s, 1].mean():+.2f}  median {np.median(bias[s, 1]):+.2f}")
    print(f"generation at the scheduled visits: data level {np.median(D[:, 0]):.2f} (90th pct {np.percentile(D[:, 0], 90):.1f}), "
          f"oscillation {np.median(D[:, 1]):.2f}")
    osc = []
    for k in ks:
        G = np.array(res[k])
        osc.append(np.median(G[:, 1]))
        print(f"   k = {k:5.3f}  level {np.median(G[:, 0]):6.2f} (rho {spearmanr(D[:, 0], G[:, 0])[0]:.2f}; 90th pct {np.percentile(G[:, 0], 90):6.1f})"
              f"   oscillation {osc[-1]:6.2f} (rho {spearmanr(D[:, 1], G[:, 1])[0]:.2f})")
    np.savez(f"{run}/gen_check_s0_{s0:g}.npz", data=D, ks=np.array(ks), **{f"k_{j}": np.array(res[k]) for j, k in enumerate(ks)})
    target = np.median(D[:, 1])
    if osc[0] >= target:
        print(f"calibrated k = 0: without diffusion the oscillation ({osc[0]:.2f}) is already >= the data ({target:.2f})")
    elif osc[-1] < target:
        print(f"calibrated k > {ks[-1]}: the oscillation stays below the data")
    else:
        j = int(np.flatnonzero(np.array(osc) >= target)[0])
        lk = lambda k: np.log(max(k, 1e-3))
        w = (target - osc[j - 1]) / (osc[j] - osc[j - 1])
        print(f"calibrated k = {np.exp(lk(ks[j - 1]) + w * (lk(ks[j]) - lk(ks[j - 1]))):.3f} (between {ks[j - 1]} and {ks[j]})")


if __name__ == "__main__":
    main()
