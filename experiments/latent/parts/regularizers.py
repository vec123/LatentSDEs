"""Regularizers: the group lasso (l2,1) of ALDSL as a proximal step on the columns of weight matrices, and the MAP prior of codes."""
import torch


@torch.no_grad()
def group_prox(weights, t, active):
    """group soft-thresholding of the columns (one group per input dimension) of each weight matrix whose flag in active is true (a modality
    without evidence in the batch is skipped: its columns would shrink without data); a column at 0 = the input dimension is not used"""
    for W, on in zip(weights, active):
        if on:
            n = W.norm(dim=0, keepdim=True)
            W.mul_((1 - t / n.clamp(min=1e-12)).clamp(min=0))


def n_used(weights):
    """the number of nonzero columns of each weight matrix"""
    return [int((W.norm(dim=0) > 0).sum()) for W in weights]


def map_prior(c, mu):
    """the MAP penalty of codes with the prior N(mu, I): 0.5 |c - mu|^2, summed"""
    return 0.5 * ((c - mu) ** 2).sum()
