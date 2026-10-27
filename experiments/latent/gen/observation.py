"""
The visit score: log1p(score) ~ Laplace(x + F, b), b = the measurement noise of near-replicate visits (floor 0.155, flare_facts.py) + the
short-term variation between visits (learned). The cut: in the window after an onset the slow state gets no gradient from the visit
(only the flare part learns there), so the slow state cannot take the flare spikes.
"""
import numpy as np
import torch
import torch.nn as nn

FLOOR = 0.155


class Observation(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.045)))))  # b = 0.20 at the start

    def scale(self):
        return FLOOR + torch.nn.functional.softplus(self.raw)

    def log_lik(self, y, x, F, win):
        x = torch.where(win, x.detach(), x)
        b = self.scale()
        return (-torch.log(2 * b) - (y - x - F).abs() / b).sum()
