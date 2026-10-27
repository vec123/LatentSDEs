"""
The posterior: a forward GRU reads the fused observations (mean, logvar) and the log gap of each step and gives q(z_t | data up to t) =
N(m_t, diag exp(lv_t)); a forward filter (no step reads a later one), so the scored states are causal.
"""
import torch
import torch.nn as nn

from experiments.latent.parts.layers import gauss


class FilterPosterior(nn.Module):
    def __init__(self, dz, hidden):
        super().__init__()
        self.gru = nn.GRU(2 * dz + 1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2 * dz)

    def forward(self, mo, lo, dt):
        """-> (m, lv, h): the posterior mean and log-variance, and the GRU state (B, N, hidden)"""
        h = self.gru(torch.cat([mo, lo, dt], -1))[0]
        m, lv = gauss(self.out(h))
        return m, lv, h
