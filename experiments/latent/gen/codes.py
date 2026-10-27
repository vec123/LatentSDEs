"""
The patient codes c_i = (level, frequency, amplitude, duration) ~ N(W x_i, diag s^2) (x_i: static features, RNAseq PCs, RNAseq presence);
q mean-field Gaussian, one sparse row per patient (only the rows of a batch move).
"""
import numpy as np
import torch
import torch.nn as nn

from experiments.latent.parts.losses import kl_normal

NAMES = ("level", "frequency", "amplitude", "duration")


class Codes(nn.Module):
    def __init__(self, n_patients, level0):
        super().__init__()
        self.m, self.l = nn.Embedding(n_patients, 4, sparse=True), nn.Embedding(n_patients, 4, sparse=True)
        nn.init.zeros_(self.m.weight)
        self.m.weight.data[:, 0] = level0
        nn.init.constant_(self.l.weight, -2.0)
        self.ls = nn.Parameter(torch.full((4,), float(np.log(0.3))))  # log s

    def sample(self, ids, draw=True):
        m = self.m(ids)
        return m + self.l(ids).exp() * torch.randn_like(m) if draw else m

    def kl(self, ids, prior_mean):
        return kl_normal(self.m(ids), self.l(ids), prior_mean, self.ls)
