"""Small layer helpers."""
import torch
import torch.nn as nn


def mlp(d_in, d_out, hidden=128, depth=2):
    """depth x (Linear, SiLU), then a Linear to d_out"""
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def soft_clamp(x, c=6.0):
    return c * torch.tanh(x / c)


def gauss(o):
    """the mean and the log-variance (soft-clamped to +-6) of a Gaussian from the two halves of the last dimension"""
    mu, lv = o.chunk(2, -1)
    return mu, soft_clamp(lv)
