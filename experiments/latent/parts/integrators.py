"""
Time integrators of latent dynamics: moment steps of N(m, v) (exact Ornstein-Uhlenbeck for a mean reversion with the bounded terms held
over the step, i.e. exponential Euler, stable for any step; Euler), sub-stepping over irregular gaps, path steps of an OU process, and the
expected log density of knots under a stationary OU prior.
"""
import numpy as np
import torch


def ou_moments(m, v, target, lam, sig2, h):
    """one exact OU step of length h towards target with rate lam and stationary variance sig2: (mean, variance, the variance the step adds)"""
    e1, e2 = torch.exp(-lam * h), torch.exp(-2 * lam * h)
    add = sig2 * (1 - e2)
    return target + (m - target) * e1, v * e2 + add, add


def euler_moments(m, v, f, g2, h):
    """one Euler step with drift f and squared diffusion g2: (mean, variance, the variance the step adds; the Jacobian of f left out)"""
    return m + f * h, v + g2 * h, g2 * h


def substeps(step, m, v, dt, max_len, scale=1.0):
    """the predict step over gaps dt (..., 1) by sub-steps of at most max_len (in the unit of dt x scale, e.g. dt in months, scale = days
    per month, max_len in days); each row does its own number of sub-steps; step(m, v, h) -> (mean, variance, ...), h in the unit of dt"""
    if dt.numel() == 0:  # e.g. a batch of patients with one step each: no transition
        return m, v
    n = torch.ceil(dt * scale / max_len).clamp(min=1)
    h = dt / n
    for k in range(int(n.max())):
        on = (k < n).float()
        m2, v2 = step(m, v, h)[:2]
        m, v = on * m2 + (1 - on) * m, on * v2 + (1 - on) * v
    return m, v


def ou_draw(x, mu, lam, sig, h, gen=None):
    """one exact OU path step of length h (rate lam, stationary sd sig)"""
    a = torch.exp(-lam * h)
    return mu + (x - mu) * a + sig * torch.sqrt(1 - a ** 2) * torch.randn(x.shape, device=x.device, generator=gen)


def ou_knots_loglik(m, ls, mu, lam, sig, nxt, first, h=1.0):
    """E_q[log density] of knots x on a grid of step h under a stationary OU prior around mu (x(0) ~ N(mu(0), sig^2)), q mean-field
    Gaussian N(m, exp(2 ls)), + the entropy of q (up to a constant); nxt: knot rows k with k + 1 in the same sequence; first: first rows"""
    v, phi, s2 = (2 * ls).exp(), torch.exp(-lam * h), sig ** 2
    e = (m[nxt + 1] - mu[nxt + 1]) - phi * (m[nxt] - mu[nxt])
    ve = s2 * (1 - phi ** 2)
    ll = (-0.5 * torch.log(2 * np.pi * ve) - (e ** 2 + v[nxt + 1] + phi ** 2 * v[nxt]) / (2 * ve)).sum()
    ll = ll + (-0.5 * torch.log(2 * np.pi * s2) - ((m[first] - mu[first]) ** 2 + v[first]) / (2 * s2)).sum()
    return ll + ls.sum()
