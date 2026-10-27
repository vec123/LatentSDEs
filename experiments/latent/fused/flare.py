"""
The flare part of the SDE model (flare_model.py is stage 1): the visit score = g(z) + F(t) (log1p units), F = the sum over the flares k up
to t of A_k exp(-(t - tau_k) / D_k). Onsets = the flare visits, a Poisson process: log rate = linear(z of the previous step, the 3
biologics, static) + sd x the frequency code (exact expectation over q(z) for the linear map). log A_k: q amortized from the GRU state at
the onset, prior linear(antibiotic, 3 biologics, static) + sd x the amplitude code. log D_k (days) = linear(same) + sd x the duration code.
The maps are linear: the coefficients are log ratios. Patient flare codes: variational, N(0, I), sparse rows. cut (days): the window after
an onset in which the score NLLs train only the flare part.
"""
import numpy as np
import torch
import torch.nn as nn

from experiments.latent.hs_e2e import MONTH
from experiments.latent.parts.flares import REACH, excess_terms
from experiments.latent.parts.layers import gauss
from experiments.latent.parts.losses import kl_normal, kl_unit
from experiments.latent.parts.masking import event_weights, lags, window


class FlarePart(nn.Module):
    def __init__(self, dz, d_s, hidden, n_codes, cut=0.0):
        super().__init__()
        self.dz, self.cut = dz, float(cut)
        self.rate, self.amp_prior, self.dur = nn.Linear(dz + 3 + d_s, 1), nn.Linear(4 + d_s, 1), nn.Linear(4 + d_s, 1)
        self.amp_q = nn.Linear(hidden, 2)
        for lin, b0 in ((self.rate, np.log(1.85)), (self.amp_prior, np.log(0.25)), (self.dur, np.log(40.0))):
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, float(b0))
        nn.init.zeros_(self.amp_q.weight)
        self.amp_q.bias.data = torch.tensor([float(np.log(0.25)), float(np.log(0.25))])
        self.ls_A = nn.Parameter(torch.tensor(-0.2))  # log sd of log A around its prior mean
        self.ls_c = nn.Parameter(torch.full((3,), float(np.log(0.3))))  # log sd of the patient effects (frequency, amplitude, duration)
        self.codes, self.codes_ls = nn.Embedding(n_codes, 3, sparse=True), nn.Embedding(n_codes, 3, sparse=True)
        nn.init.normal_(self.codes.weight, std=0.01)
        nn.init.constant_(self.codes_ls.weight, -1.0)

    def sample_codes(self, idx):
        """a draw of the flare codes of the patients idx and their KL to N(0, I), summed"""
        m, ls = self.codes(idx), self.codes_ls(idx)
        return m + ls.exp() * torch.randn_like(m), kl_unit(m, ls)

    def effects(self, cf):
        """the patient effects (frequency, amplitude, duration) = sd x the flare code"""
        return self.ls_c.exp() * cf

    def prior(self, u, s, sc):
        """(prior mean of log A, log D) for rows with the treatment flags u (..., 5) and static s; sc (..., 3) broadcast to the rows"""
        a = torch.cat([u[..., [0, 2, 3, 4]], s], -1)  # antibiotic, 3 biologics, static
        return self.amp_prior(a)[..., 0] + sc[..., 1], self.dur(a)[..., 0] + sc[..., 2]

    def log_rate(self, z, u, s, sc):
        """the log onset rate per year"""
        return self.rate(torch.cat([z, u[..., 2:5], s], -1))[..., 0] + sc[..., 0]

    def onset_prob(self, z, u, s, sc, days):
        """the probability of an onset in a step of `days` (at most one per step)"""
        return 1 - torch.exp(-torch.exp(self.log_rate(z, u, s, sc).clamp(max=6.0)) * days / 365.25)

    def terms(self, m, lv, h, ex, cf, sample=True):
        """the flare part of a batch (after encode; m, lv the posterior, h the GRU state): F_post (B, N) the excess (log1p units) at each row
        from the flares up to it (amplitudes from q, one draw or the mean); F_pred the excess before the row is seen (the earlier flares
        from q, an onset at the row with its prior mean amplitude); kl_A (sum over the onsets); rate_nll (the negative Poisson
        log-likelihood of the onsets; the onset at the first row is not counted); win the cut window"""
        on, valid, day = ex["onset"], ex["valid"], ex["day"]
        B, N = on.shape
        sc = self.effects(cf)
        pA, logD = self.prior(ex["u"], ex["s"][:, None, :].expand(B, N, -1), sc[:, None, :])
        qm, qlv = gauss(self.amp_q(h))
        qm, qlv = qm[..., 0], qlv[..., 0]
        logA = (qm + (0.5 * qlv).exp() * torch.randn_like(qm) if sample else qm).clamp(-8.0, 2.0)
        kl_A = kl_normal(qm, 0.5 * qlv, pA, self.ls_A, w=on)
        lag = lags(day)
        E = excess_terms(event_weights(on, lag, REACH), lag, logA, logD)
        F_post = E.sum(-1)
        win = window(on, lag, self.cut, valid)
        F_pred = (E * (1 - torch.eye(N, device=on.device))).sum(-1) + on * pA.clamp(-8.0, 2.0).exp()
        wz = self.rate.weight[0, :self.dz]
        lin = self.log_rate(m[:, :-1], ex["u"][:, 1:], ex["s"][:, None, :].expand(B, N - 1, -1), sc[:, None, :])
        var = (lv[:, :-1].exp() * wz ** 2).sum(-1)
        gap_y = ex["dtm"][:, 1:] * MONTH / 365.25
        rate_ll = (on[:, 1:] * lin - gap_y * torch.exp((lin + 0.5 * var).clamp(max=6.0))) * valid[:, 1:]
        return {"F_post": F_post, "F_pred": F_pred, "kl_A": kl_A, "rate_nll": -rate_ll.sum(), "logA": logA, "logD": logD, "pA": pA, "win": win}
