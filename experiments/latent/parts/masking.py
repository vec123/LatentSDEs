"""
Masks over the steps of padded sequences (B, N): causal masks, event windows, the cut (a stop-gradient under a mask), the masking of the
steps after a per-sequence cut, and the key padding of an attention over {null, views}.
"""
import torch


def causal(N, device=None):
    """(N, N) lower triangle: row t may use rows k <= t"""
    return torch.tril(torch.ones(N, N, device=device))


def lags(day):
    """(B, row t, row k) the time from row k to row t"""
    return day[:, :, None] - day[:, None, :]


def event_weights(events, lag, max_lag):
    """(B, t, k) 1 where row k is an event at or before row t (row order: a later row on the same day does not count) within max_lag"""
    N = events.shape[1]
    return events[:, None, :] * causal(N, events.device) * ((lag >= 0) & (lag <= max_lag)).float()


def window(events, lag, max_lag, valid):
    """(B, N) 1 at the rows at or up to max_lag after an event (causal)"""
    return (event_weights(events, lag, max_lag).sum(-1) > 0).float() * valid


def cut(x, mask):
    """x without gradient where mask > 0 (modular inference: those rows train only the other parts)"""
    return torch.where(mask > 0, x.detach(), x)


def mask_after(ex, cuts, keys=("valid", "has", "onset")):
    """in place: the rows after cuts[b] of each sequence b set to 0 in ex[key] for the given keys"""
    for b, c in enumerate(cuts):
        for k in keys:
            if k in ex:
                ex[k][b, int(c) + 1:] = 0


def crop_steps(ex, T, skip=("s",)):
    """the step-dimension tensors of ex cut to the first T steps (all (B, N, ...) tensors except those named in skip)"""
    N = ex["valid"].shape[1]
    return {k: (x[:, :T] if torch.is_tensor(x) and x.dim() >= 2 and x.shape[1] == N and k not in skip else x) for k, x in ex.items()}


def key_padding(presence):
    """the attention key padding mask (True = masked) over {null, views}: the null token is always a key; presence: list of (B, N, 1)"""
    return torch.cat([torch.ones_like(presence[0])] + list(presence), -1) == 0
