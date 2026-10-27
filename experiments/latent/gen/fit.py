"""
Stochastic variational inference over patient batches: the dense parameters (scales, rates, effects) with Adam, the per-patient rows
(codes, knots, flare amplitudes: sparse embeddings) with SparseAdam, so only the rows of the batch move. batch >= all patients: the full
batch. pause: seconds of rest after each step (a duty cycle for the laptop GPU: full-batch steps throttled it within ~20 s).
"""
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.latent.gen.data import gather


def summary(model):
    s, f = model.slow, model.flares
    return (f"slow: half-life {np.log(2) / s.lam().item():.1f} months, sd {s.sig().item():.3f}; visit noise {model.obs.scale().item():.3f}; "
            f"flares {f.a0.exp().item():.2f}/yr, half-life {np.log(2) * f.d0.exp().item():.0f} d, sd log A {f.ls_A.exp().item():.2f}; "
            f"patient sd (level, frequency, amplitude, duration) " + " ".join(f"{x:.2f}" for x in model.codes.ls.exp().tolist()))


def fit(model, rec, steps, batch, lr=0.03, pause=0.0, log=print, seed=0, every=200):
    dev = rec["v_y"].device
    N, n_vis = len(rec["pats"]), len(rec["v_y"])
    sparse = [m.weight for m in model.modules() if isinstance(m, nn.Embedding)]
    dense = [p for p in model.parameters() if all(p is not q for q in sparse)]
    opts = [torch.optim.Adam(dense, lr=lr), torch.optim.SparseAdam(sparse, lr=lr)]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda k: 1.0 if k < 0.6 * steps else 0.2) for o in opts]
    g = torch.Generator().manual_seed(seed)
    order, pos, t0 = torch.randperm(N, generator=g), 0, time.time()
    for step in range(steps):
        if batch >= N:
            pats = torch.arange(N)
        else:
            if pos + batch > N:
                order, pos = torch.randperm(N, generator=g), 0
            pats, pos = order[pos:pos + batch], pos + batch
        loss, parts = model.loss(gather(rec, pats.to(dev)), N, n_vis)
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o, s in zip(opts, scheds):
            o.step()
            s.step()
        if pause:
            time.sleep(pause)
        if step % every == 0 or step == steps - 1:
            log(f"{step:6d} t={time.time() - t0:5.0f}s loss {float(loss):.4f} " + " ".join(f"{k} {v:+.4f}" for k, v in parts.items())
                + f" | {summary(model)}")
