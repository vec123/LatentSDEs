"""
The SDE prior of the latent: dz = f(z, u, s) dt + g dW (u: the treatment flags of the interval, s: static covariates).

drift_mr: f = -lam (z - mu(u, s[, c])) + tanh(r(z, u, s)), lam = softplus + 0.05 per month (max 4): the mean path stays within 1 / lam of the
set point (no run-away rollouts); the diffusion as the stationary variance sig2 = g^2 / (2 lam) (start ~0.7, the scale of the latent: with g
itself, a small g and the mean reversion gave a tiny prior variance, and the KL pulled every posterior onto the set point); one step = the
exact OU step for the mean reversion with the bounded term held (exponential Euler: stable for any step). Else a free drift (Euler).
Patient code c: free (into the set-point and diffusion MLPs) or structured (level, noise, speed, response blocks, each through its own
linear map: level shifts the set point, noise the stationary variance, speed the rate lam, response the set point while a biologic is
active). z0 ~ N(z0_mean, exp(z0_logvar)). predict: one step over a gap, or sub-steps of at most `sub` days (the rollout grid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from experiments.latent.hs_e2e import MONTH
from experiments.latent.parts.integrators import euler_moments, ou_moments, substeps
from experiments.latent.parts.layers import mlp


class LatentSDE(nn.Module):
    def __init__(self, dz, d_u, d_s, d_code=0, code_struct=(), drift_mr=False, sub=0.0):
        super().__init__()
        self.code_struct, self.drift_mr, self.sub = tuple(code_struct), drift_mr, sub
        self.d_code = sum(self.code_struct) if self.code_struct else d_code
        d_free = 0 if self.code_struct else self.d_code  # the free code enters the set-point and diffusion MLPs
        if drift_mr:
            self.set_point = mlp(d_u + d_s + d_free, dz)
            self.lam_raw = nn.Parameter(torch.full((dz,), -3.0))  # softplus(-3) + 0.05 = 0.10 per month at the start (~10 months)
            if self.code_struct:
                self.w_level, self.w_noise, self.w_speed, self.w_resp = (nn.Linear(k, dz, bias=False) for k in self.code_struct)
        self.drift, self.diff = mlp(dz + d_u + d_s, dz), mlp(dz + d_u + d_s + d_free, dz)
        self.z0_mean, self.z0_logvar = nn.Parameter(torch.zeros(dz)), nn.Parameter(torch.zeros(dz))

    def lam(self, c_speed=None):
        """the mean-reversion rate per month (max 4: stable on the 7-day grid); with a structured code patient-specific"""
        raw = self.lam_raw if c_speed is None else self.lam_raw + self.w_speed(c_speed)
        return (Fn.softplus(raw) + 0.05).clamp(max=4.0)

    def split_code(self, c, u):
        if c is None:
            c = torch.zeros(*u.shape[:-1], self.d_code, device=u.device)
        return torch.split(c, list(self.code_struct), -1)

    def fg(self, z, u, s):
        """the drift and the diffusion g (the free-drift form, and the drift of drift_mr without the code)"""
        a = torch.cat([z, u, s], -1)
        g = Fn.softplus(self.diff(a) - 2.0) + 1e-3
        if self.drift_mr:
            return -self.lam() * (z - self.set_point(torch.cat([u, s], -1))) + torch.tanh(self.drift(a)), g
        return self.drift(a), g

    def step(self, m, v, u, s, h, c=None):
        """one step of length h (months) from N(m, v): (mean, variance, the variance the step adds = the noise of a rollout path); c: the
        patient code (None: the prior mean 0)"""
        if not self.drift_mr:
            f, g = self.fg(m, u, s)
            return euler_moments(m, v, f, g ** 2, h)
        a = torch.cat([m, u, s], -1)
        if self.code_struct:
            c_level, c_noise, c_speed, c_resp = self.split_code(c, u)
            lam = self.lam(c_speed)
            sig2 = Fn.softplus(self.diff(a) + self.w_noise(c_noise)) + 1e-2
            shift = self.w_level(c_level) + u[..., 1:2] * self.w_resp(c_resp)  # u[..., 1]: Biologic_active (batch.CONTROLS)
            target = self.set_point(torch.cat([u, s], -1)) + shift + torch.tanh(self.drift(a)) / lam
        else:
            cc = [c if c is not None else torch.zeros(*u.shape[:-1], self.d_code, device=u.device)] if self.d_code else []
            lam, sig2 = self.lam(), Fn.softplus(self.diff(torch.cat([a] + cc, -1))) + 1e-2
            target = self.set_point(torch.cat([u, s] + cc, -1)) + torch.tanh(self.drift(a)) / lam
        return ou_moments(m, v, target, lam, sig2, h)

    def predict(self, m, v, u, s, dt, c=None):
        """the predict step over gaps dt (months, (..., 1)) from N(m, v): one step, or sub-steps of at most `sub` days"""
        if not self.sub:
            return self.step(m, v, u, s, dt, c)[:2]
        return substeps(lambda m_, v_, h: self.step(m_, v_, u, s, h, c), m, v, dt, self.sub, scale=MONTH)
