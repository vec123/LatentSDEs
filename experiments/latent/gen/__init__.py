"""
The dedicated generator of IHS4 courses, in small parts:

    data         per-patient records (scored visits, flare onsets, visit-flare pairs, rate cells, knots) and gather(patients) -> a batch
    effects      the effect coefficients (variational, prior N(0, 1))
    codes        the patient codes (level, frequency, amplitude, duration) ~ N(W x, diag s^2)
    slow         the slow severity x(t): monthly knots with an Ornstein-Uhlenbeck prior
    flares       onsets (Poisson), amplitude per flare, exponential decay, the flare excess F(t)
    observation  the visit noise (Laplace in log1p, floor = the near-replicate noise) and the cut
    model        the generator from these parts: the ELBO of a batch of patients
    fit          stochastic variational inference over patient batches (full batch = one batch of all patients)
    simulate     courses from a start state (OU steps, drawn flares)
    evaluate     the 6-month test against the real labels
    run          the command line: python -m experiments.latent.gen.run

Model: log1p(score) ~ Laplace(x(t) + F(t), noise); trained only on the visit scores and the flare onsets up to the common cutoff
(2025-12-31; no record goes past it; the labels are never used).
"""
