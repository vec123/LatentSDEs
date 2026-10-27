"""
Patient codes of the SDE prior (auto-decoder, not amortized): one sparse row per patient, MAP (prior N(mu_c, I), penalty 0.5 |c - mu_c|^2)
or variational (mean and log-sd rows, KL to N(mu_c, I)); mu_c = 0, or with a prior network f(RNAseq, presence, static) in the last block
(the response block of a structured code).
"""
import torch
import torch.nn as nn

from experiments.latent.parts.losses import kl_unit
from experiments.latent.parts.regularizers import map_prior


class PatientCodes(nn.Module):
    def __init__(self, n, d, var=False, prior_in=0, prior_out=0):
        super().__init__()
        self.d, self.var = d, var
        self.mean = nn.Embedding(n, d, sparse=True)
        nn.init.normal_(self.mean.weight, std=0.01)
        self.log_sd = nn.Embedding(n, d, sparse=True) if var else None
        if var:
            nn.init.constant_(self.log_sd.weight, -2.0)  # sd 0.14 at the start
        self.prior = nn.Linear(prior_in, prior_out) if prior_out else None
        if self.prior is not None:
            nn.init.zeros_(self.prior.weight)
            nn.init.zeros_(self.prior.bias)

    def prior_mean(self, x):
        """(B, d): 0, or f(x) in the last block"""
        if self.prior is None:
            return torch.zeros(x.shape[0], self.d, device=x.device)
        return torch.cat([torch.zeros(x.shape[0], self.d - self.prior.out_features, device=x.device), self.prior(x)], -1)

    def sample(self, idx, prior_in):
        """the codes of the patients idx (a draw of q, or the MAP rows) and their prior term (KL or MAP penalty), summed"""
        cm, mu = self.mean(idx), self.prior_mean(prior_in)
        if self.var:
            ls = self.log_sd(idx)
            return cm + ls.exp() * torch.randn_like(cm), kl_unit(cm, ls, mu)
        return cm, map_prior(cm, mu)
