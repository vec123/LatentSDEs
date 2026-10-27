"""Prediction heads: an MLP -> (median, Laplace scale), and the input of the forecast head with its skip connections."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from experiments.latent.parts.layers import mlp


class LaplaceHead(nn.Module):
    def __init__(self, d_in, hidden=64, depth=1):
        super().__init__()
        self.net = mlp(d_in, 2, hidden, depth)

    def forward(self, h):
        o = self.net(h)
        return o[..., 0], Fn.softplus(o[..., 1]) + 1e-3


def head_input(m, lv, kind, obs=None, gru=None):
    """[posterior mean, log-variance] + skips: obs = the fused observation (mean, logvar) of the step, gru = the GRU state (no KL on either);
    + the step-kind columns"""
    return torch.cat([m, lv] + (list(obs) if obs is not None else []) + ([gru] if gru is not None else []) + [kind], -1)
