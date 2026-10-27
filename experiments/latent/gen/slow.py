"""
The slow severity x(t) (log1p units) on monthly knots, linear between them. Prior: Ornstein-Uhlenbeck around mu(t), stationary (so it can
generate): x(m + 1) - mu(m + 1) = phi (x(m) - mu(m)) + N(0, sig^2 (1 - phi^2)), phi = exp(-lam), x(0) ~ N(mu(0), sig^2). The rate lam is
bounded (half-life >= 2 months): a free rate went to 0.3 months and took the short-term variation, which belongs to the visit noise.
q: mean-field Gaussian, one sparse row per knot.
"""
import numpy as np
import torch
import torch.nn as nn

from experiments.latent.gen.data import KNOT
from experiments.latent.parts.integrators import ou_draw, ou_knots_loglik

LAM_MAX = np.log(2) / 2.0  # per month


class SlowState(nn.Module):
    def __init__(self, n_knots, init, mu0):
        super().__init__()
        self.m, self.l = nn.Embedding(n_knots, 1, sparse=True), nn.Embedding(n_knots, 1, sparse=True)
        self.m.weight.data[:, 0] = init
        nn.init.constant_(self.l.weight, -2.0)
        q = 0.10 / LAM_MAX  # rate 0.10 per month at the start
        self.lam_raw, self.ls_sig = nn.Parameter(torch.tensor(float(np.log(q / (1 - q))))), nn.Parameter(torch.tensor(float(np.log(0.4))))
        self.mu0 = nn.Parameter(torch.tensor(float(mu0)))

    def lam(self):
        return LAM_MAX * torch.sigmoid(self.lam_raw)

    def sig(self):
        return self.ls_sig.exp()

    def sample(self, ids, draw=True):
        m = self.m(ids)[:, 0]
        return m + self.l(ids)[:, 0].exp() * torch.randn_like(m) if draw else m

    @staticmethod
    def at(knots, k, frac):
        """linear interpolation of knot values at batch knot rows k (left) with the fraction frac to k + 1"""
        return (1 - frac) * knots[k] + frac * knots[k + 1]

    def log_prior(self, ids, mu, nxt, first):
        """E_q[log OU density of the knots] + the entropy of q (up to a constant); nxt: knot rows m with m + 1 of the same patient"""
        return ou_knots_loglik(self.m(ids)[:, 0], self.l(ids)[:, 0], mu, self.lam(), self.sig(), nxt, first)

    def posterior_at(self, k_glob, frac):
        """the posterior mean and sd of x at the times given by the global knot rows k_glob (left) and fractions"""
        m1, m2 = self.m.weight[k_glob, 0], self.m.weight[k_glob + 1, 0]
        v1, v2 = (2 * self.l.weight[k_glob, 0]).exp(), (2 * self.l.weight[k_glob + 1, 0]).exp()
        return (1 - frac) * m1 + frac * m2, ((1 - frac) ** 2 * v1 + frac ** 2 * v2).sqrt()

    def step(self, x, mu, days, gen=None):
        """one exact OU path step of `days`"""
        return ou_draw(x, mu, self.lam(), self.sig(), days / KNOT, gen)
