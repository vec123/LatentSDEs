"""
Decoders: the visit score from the latent (Laplace in the normalized log1p score; the measurement model adds a bias and multiplies the
scale per visit type, relative to the first type, the background visits), and the linear view decoders of the ALDSL reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from experiments.latent.parts.layers import mlp


class ScoreDecoder(nn.Module):
    def __init__(self, dz, meas=False, n_types=5):
        super().__init__()
        self.meas = meas
        self.net = mlp(dz, 2, 64, 1)
        self.vt_bias, self.vt_logs = nn.Parameter(torch.zeros(n_types)), nn.Parameter(torch.zeros(n_types))

    def forward(self, z, vt=None):
        """(mean, scale); vt None or no measurement model: g(z), the severity without visit effects"""
        o = self.net(z)
        mu, b = o[..., 0], Fn.softplus(o[..., 1]) + 1e-2
        if self.meas and vt is not None:
            mu = mu + (self.vt_bias - self.vt_bias[0])[vt]
            b = b * torch.exp((self.vt_logs - self.vt_logs[0])[vt])
        return mu, b


class ViewDecoders(nn.Module):
    def __init__(self, d_src, views):
        """views: {name: embedding width}"""
        super().__init__()
        self.dec = nn.ModuleDict({k: nn.Linear(d_src, d) for k, d in views.items()})

    def forward(self, name, x):
        return self.dec[name](x)
