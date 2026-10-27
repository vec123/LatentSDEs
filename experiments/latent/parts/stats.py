"""Small statistics helpers."""
import numpy as np


def boot(e, n=2000, seed=0):
    """the 95% bootstrap interval of the mean of e (resampling its entries, e.g. paired per-patient differences)"""
    r = np.random.default_rng(seed)
    return np.percentile(e[r.integers(0, len(e), (n, len(e)))].mean(1), [2.5, 97.5])
