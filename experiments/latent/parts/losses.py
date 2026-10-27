"""Losses and divergences."""
import torch


def laplace_nll(target, mu, b, w):
    """the Laplace NLL (without log 2) of target given (mu, b), weighted mean over w (a mask or weights)"""
    return (((target - mu).abs() / b + b.log()) * w).sum() / w.sum().clamp(min=1)


def laplace_terms(t, mu, b, w=None):
    """a prediction head: the L1 of the median and the Laplace NLL of the scale (the median held fixed), weighted means"""
    w = torch.ones_like(t) if w is None else w
    s = w.sum().clamp(min=1e-6)
    return (w * (t - mu).abs()).sum() / s, (w * ((t - mu.detach()).abs() / b + b.log())).sum() / s


def gauss_kl(m_q, lv_q, m_p, v_p):
    """KL(N(m_q, exp(lv_q)) || N(m_p, v_p)) of diagonal Gaussians, summed over the last dimension"""
    return 0.5 * (v_p.log() - lv_q + (lv_q.exp() + (m_q - m_p) ** 2) / v_p - 1).sum(-1)


def gauss_kl_logvar(m_q, lv_q, m_p, lv_p):
    """KL(N(m_q, exp(lv_q)) || N(m_p, exp(lv_p))) of diagonal Gaussians, summed over the last dimension"""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (m_q - m_p) ** 2) / lv_p.exp() - 1).sum(-1)


def kl_normal(m, ls, pm, ps, w=None):
    """KL(N(m, exp(2 ls)) || N(pm, exp(2 ps))), summed (ls, ps: log standard deviations; w: optional weights of the terms)"""
    k = ps - ls + ((2 * ls).exp() + (m - pm) ** 2) / (2 * (2 * ps).exp()) - 0.5
    return (k if w is None else w * k).sum()


def kl_unit(m, ls, mu=0.0):
    """KL(N(m, exp(2 ls)) || N(mu, 1)), summed"""
    return 0.5 * ((2 * ls).exp() + (m - mu) ** 2 - 1 - 2 * ls).sum()


def poisson_ll(log_rate_events, log_rate_cells, dt, max_log=6.0):
    """the log-likelihood of a Poisson process: sum of the log rates at the events - the integral of the rate (piecewise constant on
    cells of length dt; the log rate clamped at max_log)"""
    return log_rate_events.sum() - (dt * log_rate_cells.clamp(max=max_log).exp()).sum()


def view_recon(codes, decs, b, hidden_too=False):
    """ALDSL view reconstruction (x^v ~ P^v h): the mean squared error of each decoded code against the clean standardized embedding, at the
    steps where a sample is new (each sample once): only shown samples (a modality code), or with hidden_too also the samples that modality
    dropout hid (the shared latent: it must then predict them from the EHR, the other modality and the history); returns (image, RNAseq)"""
    out = []
    for c, dec, k in zip(codes, decs, ("img", "rna")):
        n = b[f"{k}_a" if hidden_too else f"{k}_n"]
        e = ((dec(c) - b[f"{k}_t"]) ** 2).mean(-1)
        out.append((e * n).sum() / n.sum().clamp(min=1))
    return out
