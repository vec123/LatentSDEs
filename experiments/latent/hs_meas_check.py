"""
Check of the measurement-model hypothesis on a fused SDE run with --meas 1 (no training; CPU, under a minute):
the real label = the severity without visit effects g(z), the visit scores = g(z) + the visit-type effect relative to background visits.

A. scored visits of the training patients per visit type: share and mean score.
B. g(z) decoded from the posterior mean at a step (no rollout), against the label of the step: the real label at the final step (held-out)
   and the visit-scale derived label at the quarter ends (held-out). If g(z) is unbiased on the derived labels but ~1.4 above the real
   labels, the visit types (relative to background) do not explain the offset of the real label.
C. the SDE rollout from the final step against g(z) at the final step: the drift of the 6-month forecast.

    python -m experiments.latent.hs_meas_check runs/hs_fused_sde_<stamp>
"""
import sys

import numpy as np
import torch

from experiments.latent.hs_e2e import PATHS, QUERY, VISIT, Data
from experiments.latent.hs_fused_sde import make_model, prepare, rollout_forecast

TYPES = ["background", "flare_reactive", "follow_up", "treatment_monitoring"]


def g_at(model, data, rows):
    """g(z) in score units from the posterior mean at the given rows"""
    pats = np.unique(data.prow[rows])
    out = {}
    with torch.no_grad():
        for a in range(0, len(pats), 50):
            sel = list(pats[a:a + 50])
            model.encode(data.pack(sel, "cpu"))
            mu = model.decode(model.last_post[0])[0].numpy()
            for b, i in enumerate(sel):
                for k in range(data.off[i + 1] - data.off[i]):
                    out[data.off[i] + k] = mu[b, k]
    mu = np.array([out[r] for r in rows])
    return np.expm1(mu * data.stats["ls"] + data.stats["lm"])


def main():
    run = sys.argv[1]
    torch.set_num_threads(4)
    ck = torch.load(f"{run}/model.pt", map_location="cpu")
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    rng = np.random.default_rng(ck["args"].get("seed", 0))
    pool = rng.permutation(np.flatnonzero(~data.flag))
    k = int(0.1 * len(pool))
    data.train_idx, data.test_idx = np.sort(pool[k:]), np.flatnonzero(data.flag)
    model = make_model(data, ck["args"])
    model.load_state_dict(ck["model"])
    model.eval()

    print("A. scored visits of the training patients: share and mean score per visit type")
    sv = data.has_score & np.isin(data.prow, data.train_idx)
    for c, name in enumerate(TYPES):
        s = sv & (data.vt == c)
        print(f"   {name:22s} {s.sum() / sv.sum():6.1%}  mean score {data.score[s].mean():6.2f}")

    te = data.test_idx
    fr = np.array([data.final_row(i) for i in te])
    q = np.flatnonzero(np.isin(data.prow, te) & (data.kind == QUERY) & data.ym)
    q = np.sort(np.random.default_rng(1).choice(q, min(5000, len(q)), replace=False))
    gf, gq = g_at(model, data, fr), g_at(model, data, q)
    print("\nB. g(z) from the posterior mean (no rollout), held-out patients: mean error (g - label) and MAE")
    print(f"   final step, real label         n={len(fr):5d}  mean error {np.mean(gf - data.y[fr]):+.3f}  MAE {np.abs(gf - data.y[fr]).mean():.3f}")
    print(f"   quarter ends, derived label    n={len(q):5d}  mean error {np.mean(gq - data.y[q]):+.3f}  MAE {np.abs(gq - data.y[q]).mean():.3f}")
    vq = np.flatnonzero(np.isin(data.prow, te) & (data.kind == VISIT) & data.has_score)
    vq = np.sort(np.random.default_rng(2).choice(vq, min(5000, len(vq)), replace=False))
    gv = g_at(model, data, vq)
    for c, name in enumerate(TYPES):
        s = data.vt[vq] == c
        if s.sum():
            print(f"   visit score (q has seen it), {name:20s} n={s.sum():5d}  mean error {np.mean(gv[s] - data.score[vq][s]):+.3f}")

    gen = torch.Generator().manual_seed(0)
    roll = rollout_forecast(model, data, [(i, data.final_row(i) - data.off[i]) for i in te], "cpu", gen=gen)
    print(f"\nC. 6-month SDE rollout from the final step - g(z) at the final step: mean {np.mean(roll - gf):+.3f}, "
          f"median {np.median(roll - gf):+.3f}; rollout - real label {np.mean(roll - data.y[fr]):+.3f}")


if __name__ == "__main__":
    main()
