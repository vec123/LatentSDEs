"""
The fused-latent SDE model of the HS challenge, in components (hs_fused_sde.py is the command line and re-exports the old names):

    batch       the SDE arrays of the data (prepare), the batch extras (extras), the rollout-loss batches (roll_batch)
    encoders    FusionEncoder: modality tokens, cross-attention over {null, views}, the fused observation (mean, logvar)
    posterior   FilterPosterior: a forward GRU filter q(z_t | data up to t)
    sde         LatentSDE: the prior dynamics (mean-reverting or free drift, diffusion, patient codes), step and predict
    decoders    ScoreDecoder (g(z) and the visit-type measurement model), ViewDecoders (ALDSL reconstruction)
    heads       LaplaceHead (median, scale) and the head input
    codes       PatientCodes: auto-decoder codes of the SDE prior (MAP or variational, optional RNAseq prior)
    flare       FlarePart: onsets (Poisson), amortized amplitudes, decay, flare codes, the flare terms of a batch
    model       FusedSDE (the components together; the old attribute names; old checkpoints load) and make_model
    objective   score_nll, sde_terms, flare_terms, rollout_loss
    inference   fit_codes: the codes of new patients
    train       the training loop
    evaluate    rollout_forecast, flare_report, evaluate_sde
"""
