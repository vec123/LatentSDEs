"""
Route A, step 1: the GRU states of a run (hs_e2e.py) for all steps of the train folder, reduced to 32 whitened principal components, and
the transitions between visit steps (cached for sde_fit.py). The GRU is not changed: the forecasts of head 1 and head 2 stay as they are.

    python -m experiments.latent.sde_states runs/hs_e2e_<stamp>        # about 2 minutes on the GPU
"""
import argparse
import time

import numpy as np
import torch

from experiments.latent.hs_e2e import VISIT, Data, MONTH, PATHS
from experiments.latent.hs_task2 import load

D_Z = 32


def active_drug(st, en, dr, day):
    """drug (0..2) of the biologic active on a day (the latest start), -1 for none"""
    act = (st <= day) & (en >= day)
    return int(dr[np.flatnonzero(act)[np.argmax(st[act])]]) if act.any() else -1


def plan_of_span(st, en, dr, a, b):
    """plan index (0 none, 1..3 drug + 1) for the span (a, b], or -1 if a biologic starts or stops inside it"""
    if ((st > a) & (st <= b)).any() or ((en >= a) & (en < b)).any():
        return -1
    return active_drug(st, en, dr, b) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, ck = load(args.run, device)
    t0 = time.time()
    H = np.zeros((len(data.y), 128), np.float16)
    order = np.argsort([data.n_rows(i) for i in range(data.n_pat)])
    with torch.no_grad():
        for a in range(0, len(order), 50):
            sel = [int(i) for i in order[a:a + 50]]
            h = model.encode(data.pack(sel, device)).cpu().numpy()
            for b, i in enumerate(sel):
                H[data.off[i]:data.off[i + 1]] = h[b, :data.n_rows(i)]
            if a % 5000 == 0:
                print(f"{a}/{data.n_pat} patients, {time.time() - t0:.0f}s", flush=True)
    train = ~data.flag[data.prow]
    Hf = H.astype(np.float32)
    mu = Hf[train].mean(0)
    U, S, Vt = np.linalg.svd(Hf[train][::5] - mu, full_matrices=False)
    comps = Vt[:D_Z]
    sd = (S[:D_Z] / np.sqrt(len(U) - 1))
    Z = ((Hf - mu) @ comps.T / sd).astype(np.float32)
    print(f"{D_Z} components hold {np.sum(S[:D_Z] ** 2) / np.sum(S ** 2):.3f} of the state variance")
    # transitions between visit steps, j visits apart, with a constant plan; per patient
    pairs = []
    for i in range(data.n_pat):
        s0, s1 = data.off[i], data.off[i + 1]
        vr = s0 + np.flatnonzero(data.kind[s0:s1] == VISIT)
        st, en, dr = data.bio[i]
        for j in (1, 2, 3, 5):
            for a_, b_ in zip(vr[:-j], vr[j:]):
                dt = (data.day[b_] - data.day[a_]) / MONTH
                if dt < 1 / 30.44:
                    continue
                p = plan_of_span(st, en, dr, data.day[a_], data.day[b_])
                if p >= 0:
                    pairs.append((i, a_, b_, dt, p))
    P = np.array(pairs, dtype=np.float64)
    print(f"{len(P)} transitions", flush=True)
    np.savez(f"{args.run}/sde_cache.npz", Z=Z, mu=mu, comps=comps, sd=sd, pairs=P)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
