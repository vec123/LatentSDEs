"""
The 6-month test: from the posterior at the cutoff, n_paths courses per patient for 6 months (treatment held); per course the time mean
of expm1(x + F); point = the median over the courses (MAE-optimal; the mean also printed); 80% interval from the courses; against the real
label (mean IHS4 of Jan-Jun 2026), held-out patients and the training pool (the generator never sees a label).
"""
import numpy as np
import pandas as pd
import torch

from experiments.latent.gen.simulate import generate


@torch.no_grad()
def six_month_test(model, rec, log, n_paths=64, chunk=1000, seed=0, out=None):
    dev = rec["v_y"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    res = {}
    for name, mask in (("held-out", rec["held"]), ("training pool", ~rec["held"])):
        idx = np.flatnonzero(mask & np.isfinite(rec["label"]))
        Q = []
        for a in range(0, len(idx), chunk):
            p = torch.as_tensor(idx[a:a + chunk], device=dev)
            g, _ = generate(model, rec, p, rec["T"][p], 6.0, n_paths, gen=gen)
            avg = torch.expm1(g).clamp(min=0).mean(-1)
            Q.append(torch.stack([avg.median(1).values, avg.mean(1), avg.quantile(0.1, 1), avg.quantile(0.9, 1)], 1).cpu())
        med, mean, lo, hi = torch.cat(Q).numpy().T
        y = rec["label"][idx]
        log(f"6-month test, {name} (n={len(idx)}): median of the courses MAE {np.abs(med - y).mean():.3f} (mean error {np.mean(med - y):+.3f}, "
            f"Spearman {pd.Series(med).corr(pd.Series(y), method='spearman'):.3f}); mean of the courses MAE {np.abs(mean - y).mean():.3f}; "
            f"80% interval covers {np.mean((y >= lo) & (y <= hi)):.1%}")
        res[name] = dict(patient_id=rec["pats"][idx], y=y, median=med, mean=mean, lo=lo, hi=hi)
    if out:
        np.savez(out, **res["held-out"])
    return res
