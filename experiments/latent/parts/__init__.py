"""
Generic building blocks of the latent models (model-independent, reused by fused/ and gen/):

    layers        mlp, soft_clamp, gauss (split a Gaussian output into mean and log-variance)
    losses        Laplace NLL and head terms, Gaussian KLs, the Poisson process log-likelihood, view reconstruction
    integrators   exact Ornstein-Uhlenbeck and Euler moment steps, sub-stepping over gaps, OU path steps, the OU prior of knots
    masking       causal masks, event windows, the cut (stop-gradient under a mask), masking of steps after a cut
    regularizers  the group lasso (l2,1) proximal step, the MAP code prior
    flares        the excess of decaying flares (padded batches or visit-flare pairs), one rollout step of the flare excess
    stats         the bootstrap interval of a mean
"""
