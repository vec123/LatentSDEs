"""
Courses from a start state: the slow state drawn from its posterior at the start (start_mean: its mean), exact OU steps on a 7-day grid
around the level of the held treatment; the observed flares up to the start decay (posterior mean amplitudes, own durations); new onsets
drawn from the rate (Bernoulli per step), amplitudes from the prior (antibiotic at its data share), decay with the duration of the held
treatment. The result is the log1p severity x + F without visit noise (for visit scores add the noise of observation.py).
"""
import torch

from experiments.latent.flare_model import REACH
from experiments.latent.gen.data import KNOT, gather
from experiments.latent.gen.flares import YEAR


@torch.no_grad()
def generate(model, rec, pats, t_start, months, n_paths, gen=None, start_mean=False, u_held=None, step_days=7.0):
    """(B, n_paths, steps) log1p severity and the grid (days after the start); pats LongTensor, t_start (B,) days from the first visit,
    u_held (B, 3) the biologics (default: those at the cutoff)"""
    dev = rec["v_y"].device
    b = gather(rec, pats)
    B, bp = b["B"], torch.arange(b["B"], device=dev)
    th = model.effects.sample(False)
    c = model.codes.sample(pats, False)
    rna = b["R"] @ th["g_R"]
    u = rec["uT"][pats] if u_held is None else u_held
    mu = model.level(u, c, th, rna, bp)
    m0, s0 = model.slow.posterior_at(rec["off_k"][pats] + (t_start // KNOT).long(), (t_start % KNOT) / KNOT)
    x = m0[:, None] + (0.0 if start_mean else s0[:, None] * torch.randn((B, n_paths), device=dev, generator=gen))
    _, logD_obs = model.flares.prior(b["f_u"], b["f_ab"], c, th, rna, b["f_p"])
    lag0 = t_start[b["f_p"]] - b["f_t"]
    keep = (lag0 >= 0) & (lag0 <= REACH)
    fp, flag = b["f_p"][keep], lag0[keep]
    fA, fD = model.flares.sample_amp(b["f_id"], False)[keep].clamp(-8, 2).exp(), logD_obs[keep].exp()
    pA, logD = model.flares.prior(u, torch.full((B,), float(model.flares.v_mean[3]), device=dev), c, th, rna, bp)
    decay, sA = torch.exp(-step_days / logD.exp())[:, None], model.flares.ls_A.exp()
    steps = int(round(months * KNOT / step_days))
    Fn = torch.zeros((B, n_paths), device=dev)
    out = torch.zeros((B, n_paths, steps), device=dev)
    for k in range(steps):
        s = (k + 1) * step_days
        x = model.slow.step(x, mu[:, None], step_days, gen)
        p_on = 1 - torch.exp(-model.flares.log_rate(t_start + s, u, c, th, rna, bp).clamp(max=6).exp() * step_days / YEAR)
        on = (torch.rand((B, n_paths), device=dev, generator=gen) < p_on[:, None]).float()
        A = torch.exp((pA[:, None] + sA * torch.randn((B, n_paths), device=dev, generator=gen)).clamp(-8, 2))
        Fn = Fn * decay + on * A
        Fo = torch.zeros(B, device=dev).index_add_(0, fp, fA * torch.exp(-(flag + s) / fD))
        out[:, :, k] = x + Fn + Fo[:, None]
    return out, (torch.arange(steps, device=dev) + 1) * step_days
