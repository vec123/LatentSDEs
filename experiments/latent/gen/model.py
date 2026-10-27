"""
The generator from its parts, and the ELBO of a batch of patients: the patient terms (visits, onsets, knots, amplitudes, codes) are sums
over the batch and are scaled by N / B (an unbiased estimate of the full sums); the KL of the effects counts once.
"""
import torch
import torch.nn as nn

from experiments.latent.gen.codes import Codes
from experiments.latent.gen.effects import Effects
from experiments.latent.gen.flares import Flares, _bio
from experiments.latent.gen.observation import Observation
from experiments.latent.gen.slow import SlowState


class Generator(nn.Module):
    def __init__(self, rec):
        super().__init__()
        level = rec["k_init"][rec["off_k"][:-1]].cpu()  # the patient's median log1p visit score
        self.effects = Effects(rec["X"].shape[1])
        self.codes = Codes(len(rec["pats"]), level - level.mean())
        self.slow = SlowState(len(rec["k_init"]), rec["k_init"].cpu(), level.mean())
        self.flares = Flares(len(rec["f_t"]), torch.cat([rec["f_u"], rec["f_ab"][:, None]], 1).mean(0).cpu())
        self.obs = Observation()

    def level(self, u, c, th, rna, p):
        """the mean mu of the slow state with the biologics u, for the batch patients p"""
        return self.slow.mu0 + c[p, 0] + u @ th["b_L"] + _bio(u) * rna[p, 0]

    def terms(self, b, draw=True):
        """the patient terms of a batch (sums over its patients): visit log-likelihood, onset log-likelihood, knot prior, amplitude KL,
        code KL"""
        th = self.effects.sample(draw)
        c = self.codes.sample(b["pats"], draw)
        rna = b["R"] @ th["g_R"]
        mu = self.level(b["k_u"], c, th, rna, b["k_p"])
        x = self.slow.at(self.slow.sample(b["k_id"], draw), b["v_k"], b["v_f"])
        pA, logD = self.flares.prior(b["f_u"], b["f_ab"], c, th, rna, b["f_p"])
        F = self.flares.excess(len(b["v_y"]), b["q_j"], b["q_k"], b["q_lag"], self.flares.sample_amp(b["f_id"], draw), logD)
        return {"visits": self.obs.log_lik(b["v_y"], x, F, b["v_win"]), "onsets": self.flares.rate_ll(b, c, th, rna),
                "knots": self.slow.log_prior(b["k_id"], mu, b["k_next"], b["k_first"]),
                "kl_amplitude": self.flares.kl_amp(b["f_id"], pA), "kl_code": self.codes.kl(b["pats"], b["X"] @ th["W"].T)}

    def loss(self, b, n_patients, n_visits):
        """- ELBO / (all scored visits), and the terms per scored visit (scaled to all patients)"""
        t = self.terms(b)
        w = n_patients / b["B"]
        local = t["visits"] + t["onsets"] + t["knots"] - t["kl_amplitude"] - t["kl_code"]
        return -(w * local - self.effects.kl()) / n_visits, {k: float(v) * w / n_visits for k, v in t.items()}
