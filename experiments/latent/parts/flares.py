"""
Flare mathematics: the excess F of flares that decay exponentially, A exp(-(t - tau) / D), from the event weights of a padded batch or
from visit-flare pairs; one rollout step of the flare excess (drawn onsets, or the expected burden).
"""
import torch

REACH = 270.0  # days: a flare older than this adds nothing (exp(-270 / 41) < 0.002)


def excess_terms(w, lag, logA, logD):
    """(B, t, k) the excess at row t from the onset at row k (w: event weights of parts.masking; logA, logD (B, N) of the onset rows)"""
    return w * torch.exp(logA[:, None, :] - lag.clamp(min=0) / logD.exp()[:, None, :])


def excess_pairs(n, j, k, lag, logA, logD):
    """(n,) the excess at n visits from visit-flare pairs (visit j, onset k, lag = visit time - onset time >= 0)"""
    return torch.zeros(n, device=logA.device).index_add_(0, j, (logA[k].clamp(-8, 2) - lag / logD[k].exp()).exp())


def rollout_step(F, decay, p_on, pA, sA, draw=True, gen=None):
    """one step of the flare excess: decay, then new onsets with probability p_on per step (at most one), log amplitude ~ N(pA, sA^2);
    draw False: the expected burden p_on E[A]"""
    if draw:
        on = (torch.rand(p_on.shape, device=p_on.device, generator=gen) < p_on).float()
        return F * decay + on * torch.exp((pA + sA * torch.randn(p_on.shape, device=p_on.device, generator=gen)).clamp(-8.0, 2.0))
    return F * decay + p_on * torch.exp(pA + 0.5 * sA ** 2)
