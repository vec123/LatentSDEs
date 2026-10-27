"""
Is the shape of the free rollouts systematic? (no training; CPU, ~1 min per model.) 150 random training patients with >= 24 months of
record: one deterministic rollout per patient from the posterior MEAN at the first visit (no sampled start state, no diffusion; flare
models: + the expected flare burden), the patient's code and own treatment path, 36 months. Per bin of months since the first visit:
  shape       the mean of (rollout - rollout at month 0) against the mean of (visit score - first visit score) of the data
  error       the mean and median of (rollout - visit score) at the scored visits (all visits; the scheduled ones in brackets)
A rise and fall in the first months of the rollout that the data does not have is an artifact of the drift (the SDE is trained on steps
between visits, not on long free runs).
--gen: also the generation check at the scheduled visits (background, treatment monitoring) of the first 36 months, patients with >= 4:
5 generated courses per patient (start state drawn from the posterior; flare models: drawn flares; + visit noise Laplace 0.155 in log1p,
the near-replicate value of flare_facts.py), without and with the diffusion of z; per patient the level (mean) and the oscillation
(mean |change| between successive scheduled visits), data against generated: medians over the patients, Spearman over the patients.

    python -m experiments.latent.rollout_shape runs/hs_fused_sde_<stamp> [more runs] [--gen]
"""
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.latent.hs_e2e import MONTH
from experiments.latent.hs_fused_sde import extras, flare_terms
from experiments.latent.plot_code_archetypes import load
from experiments.latent.plot_code_rollouts import rollouts

BINS = [0, 1, 3, 6, 9, 12, 18, 24, 36.01]


def main():
    torch.set_num_threads(2)
    gen_check = "--gen" in sys.argv
    for run in [a for a in sys.argv[1:] if not a.startswith("--")]:
        G = []
        ck, data, model = load(run)
        pool = np.random.default_rng(ck["args"].get("seed", 0)).permutation(np.flatnonzero(~data.flag))
        tr = np.sort(pool[int(0.1 * len(pool)):])
        span = np.array([(data.day[data.off[i + 1] - 1] - data.day[data.off[i]]) / MONTH for i in tr])
        pats = [int(i) for i in np.random.default_rng(1).choice(tr[span >= 24], 150, replace=False)]
        gen = torch.Generator().manual_seed(0)
        rows = []
        with torch.no_grad():
            for a in range(0, len(pats), 50):
                sel = pats[a:a + 50]
                model.encode(data.pack(sel, "cpu"))
                m = model.last_post[0]
                F0 = (flare_terms(model, extras(data, sel, "cpu"), model.pflare.weight[sel].detach(), sample=False)["F_post"][:, 0].numpy()
                      if model.flare else np.zeros(len(sel)))
                for b, i in enumerate(sel):
                    grid, p = rollouts(model, data, i, m[b, 0], torch.full_like(m[b, 0], -30.0), 1, False, gen, F0[b], months=36)
                    s_ = slice(data.off[i], data.off[i + 1])
                    h = data.has_score[s_]
                    t, sc = data.day[s_][h], data.score[s_][h]
                    sched = np.isin(data.vt[s_][h], (0, 3))
                    keep = t - data.day[s_][0] <= 36 * MONTH
                    at = np.clip(np.searchsorted(grid, t[keep]), 0, len(grid) - 1)
                    mo = (t[keep] - data.day[s_][0]) / MONTH
                    for k in range(keep.sum()):
                        rows.append((mo[k], p[0, at[k]] - p[0, 0], sc[keep][k] - sc[0], p[0, at[k]] - sc[keep][k], sched[keep][k]))
                    js = np.flatnonzero(sched[keep])
                    if gen_check and len(js) >= 4:
                        y = sc[keep][js]
                        out = [y.mean(), np.abs(np.diff(y)).mean()]
                        for zd in (False, True):
                            gr, q = rollouts(model, data, i, m[b, 0], model.last_post[1][b, 0], 5, True, gen, F0[b], zdiff=zd, months=36)
                            a2 = np.clip(np.searchsorted(gr, t[keep][js]), 0, len(gr) - 1)
                            g = np.expm1(np.log1p(q[:, a2]) + np.random.default_rng(i).laplace(0, 0.155, (5, len(js)))).clip(min=0)
                            out += [g.mean(1).mean(), np.abs(np.diff(g, axis=1)).mean(1).mean()]
                        G.append(out)
        r = np.array(rows, dtype=float)
        print(f"\n{run}: {len(pats)} patients, {len(r)} scored visits in the first 36 months (deterministic rollout from the first visit)")
        print(f"   {'months':>10s} {'n':>6s} {'rollout - month 0':>18s} {'data - first visit':>19s} {'rollout - data mean / median':>30s} {'(scheduled)':>12s}")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (r[:, 0] >= lo) & (r[:, 0] < hi)
            ss = s & (r[:, 4] > 0)
            if s.sum():
                print(f"   {f'{lo:g}-{hi:.0f}':>10s} {s.sum():6d} {r[s, 1].mean():+18.2f} {r[s, 2].mean():+19.2f} "
                      f"{r[s, 3].mean():+14.2f} / {np.median(r[s, 3]):+6.2f}   {r[ss, 3].mean() if ss.sum() else np.nan:+10.2f}")
        if G:
            G = np.array(G)
            sp = lambda a, b_: spearmanr(a, b_)[0]
            print(f"   generation at the scheduled visits ({len(G)} patients with >= 4 in 36 months; medians over the patients; Spearman with "
                  f"the data over the patients)")
            print(f"      data                                 level {np.median(G[:, 0]):6.2f}   oscillation {np.median(G[:, 1]):6.2f}")
            for k, name in ((2, "without diffusion (drift + drawn flares)"), (4, "with diffusion (+ drawn flares)")):
                print(f"      {name:38s} level {np.median(G[:, k]):6.2f} (rho {sp(G[:, 0], G[:, k]):.2f})   oscillation "
                      f"{np.median(G[:, k + 1]):6.2f} (rho {sp(G[:, 1], G[:, k + 1]):.2f});  90th pct of the level {np.percentile(G[:, k], 90):7.1f} "
                      f"(data {np.percentile(G[:, 0], 90):.1f})")


if __name__ == "__main__":
    main()
