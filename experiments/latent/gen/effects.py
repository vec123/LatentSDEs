"""
The effect coefficients: mean-field Gaussian q, prior N(0, 1) (log-scale effects). Blocks:
    b_L   biologics (adalimumab, bimekizumab, secukinumab) on the level of the slow state
    b_nu  biologics on the flare rate
    b_A   biologics + antibiotic within 7 days of the onset on log A (centred over the onsets)
    b_D   the same on log D
    g_R   RNAseq PC1-3 x biologic on (level, rate, amplitude, duration)
    W     features -> the prior means of the patient codes (level, frequency, amplitude, duration)
"""
import torch
import torch.nn as nn

from experiments.latent.parts.losses import kl_unit

BLOCKS = {"b_L": 3, "b_nu": 3, "b_A": 4, "b_D": 4, "g_R": 12}


class Effects(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.px = n_features
        n = sum(BLOCKS.values()) + 4 * n_features
        self.m, self.l = nn.Parameter(torch.zeros(n)), nn.Parameter(torch.full((n,), -3.0))

    def _split(self, th):
        out, a = {}, 0
        for k, n in BLOCKS.items():
            out[k], a = th[a:a + n], a + n
        out["g_R"] = out["g_R"].view(3, 4)
        out["W"] = th[a:].view(4, self.px)
        return out

    def sample(self, draw=True):
        return self._split(self.m + self.l.exp() * torch.randn_like(self.m) if draw else self.m)

    def sd(self):
        return self._split(self.l.exp().detach())

    def kl(self):
        return kl_unit(self.m, self.l)
