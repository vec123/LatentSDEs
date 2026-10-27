"""
The flare part: onsets = the flare visits, a Poisson process (exact log-likelihood on monthly cells): log rate = a0 + g_t years +
c_freq + b_nu . biologics + biologic g_nu . rna; per flare log A ~ N(mu_A + c_amp + b_A . V + biologic g_A . rna, sig_A) (q: one sparse
row per onset), V = [biologics, antibiotic within 7 days] centred over the onsets; log D = d0 + c_dur + b_D . V + biologic g_D . rna (days);
the excess F(t) = sum over the onsets k with tau_k <= t of A_k exp(-(t - tau_k) / D_k).
"""
import numpy as np
import torch
import torch.nn as nn

from experiments.latent.parts.flares import excess_pairs
from experiments.latent.parts.losses import kl_normal, poisson_ll

YEAR = 365.25


def _bio(u):
    return (u.sum(1) > 0).float()


class Flares(nn.Module):
    def __init__(self, n_onsets, v_mean):
        super().__init__()
        t = lambda x: nn.Parameter(torch.tensor(float(x)))
        self.a0, self.g_t, self.mu_A, self.ls_A, self.d0 = t(np.log(1.85)), t(0.0), t(np.log(0.25)), t(-0.2), t(np.log(40.0))
        self.register_buffer("v_mean", v_mean)
        self.m, self.l = nn.Embedding(n_onsets, 1, sparse=True), nn.Embedding(n_onsets, 1, sparse=True)
        nn.init.constant_(self.m.weight, float(np.log(0.25)))
        nn.init.constant_(self.l.weight, -1.0)

    def log_rate(self, t, u, c, th, rna, p):
        """log rate per year at times t (days from the first visit) with the biologics u, for the batch patients p"""
        return self.a0 + self.g_t * t / YEAR + c[p, 1] + u @ th["b_nu"] + _bio(u) * rna[p, 1]

    def rate_ll(self, b, c, th, rna):
        on = self.log_rate(b["f_t"], b["f_u"], c, th, rna, b["f_p"])
        return poisson_ll(on, self.log_rate(b["c_t"], b["c_u"], c, th, rna, b["c_p"]), b["c_dt"] / YEAR)

    def prior(self, u, ab, c, th, rna, p):
        """the prior mean of log A and log D for onsets with the biologics u, the antibiotic flag ab, of the batch patients p"""
        V = torch.cat([u, ab[:, None]], 1) - self.v_mean
        bio = _bio(u)
        return self.mu_A + c[p, 2] + V @ th["b_A"] + bio * rna[p, 2], self.d0 + c[p, 3] + V @ th["b_D"] + bio * rna[p, 3]

    def sample_amp(self, ids, draw=True):
        m = self.m(ids)[:, 0]
        return m + self.l(ids)[:, 0].exp() * torch.randn_like(m) if draw else m

    def kl_amp(self, ids, prior_A):
        return kl_normal(self.m(ids)[:, 0], self.l(ids)[:, 0], prior_A, self.ls_A)

    @staticmethod
    def excess(n_visits, q_j, q_k, lag, logA, logD):
        """F at the visits from the visit-flare pairs (lag = visit time - onset time >= 0, days)"""
        return excess_pairs(n_visits, q_j, q_k, lag, logA, logD)
