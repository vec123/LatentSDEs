"""The patient codes of new patients (auto-decoder, not amortized): a small optimization per patient with the model frozen."""
import numpy as np
import torch

from experiments.latent.fused.batch import extras
from experiments.latent.fused.objective import sde_terms
from experiments.latent.parts.masking import crop_steps, mask_after


def fit_codes(model, data, pats, device, steps=100, lr=0.05, chunk=25, beta=1.0, w_pred=1.0, seed=0, cuts=None):
    """the codes of new patients: only the codes optimized (Adam, from the prior mean) against the terms of the SDE prior that they enter,
    on each patient's data up to the cutoff: beta x (filtering KL + code prior 0.5 |c - mu_c|^2, per step) + w_pred x predictive score NLL;
    chunks of patients of similar length; cuts: per entry of pats the last local row to use (the steps after it are masked; the posterior
    is a forward filter, so the steps up to the cut read no later data); a patient may appear more than once with different cuts; with the
    flare part also the flare codes (prior N(0, I)) against the score NLL, the amplitude KL and the onset NLL; returns (len(pats),
    d_code [+ 3]): [code | flare code]"""
    torch.manual_seed(seed)
    pats = np.asarray(pats)
    has_flare = model.flare is not None
    order = np.argsort(np.asarray(cuts) if cuts is not None else [data.n_rows(int(i)) for i in pats], kind="stable")
    out = np.zeros((len(pats), model.d_code + 3 * has_flare), np.float32)
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for a in range(0, len(pats), chunk):
        idx = order[a:a + chunk]
        sel = [int(i) for i in pats[idx]]
        with torch.no_grad():
            model.encode(data.pack(sel, device))
        ex = extras(data, sel, device, prior_rows=[data.off[i] + int(np.asarray(cuts)[j]) for i, j in zip(sel, idx)] if cuts is not None
                    else None)
        if cuts is not None:
            mask_after(ex, [cuts[j] for j in idx])
            T = int(np.asarray(cuts)[idx].max()) + 1  # drop the steps after the last cut of the chunk (less work)
            ex = crop_steps(ex, T)
            model.last_post = tuple(t[:, :T] for t in model.last_post)
            model.last_h = model.last_h[:, :T]
        with torch.no_grad():
            mu_c = model.code_prior_mean(ex["cp_in"]) if model.d_code else None
        c = mu_c.clone().requires_grad_(True) if model.d_code else None
        cf = torch.zeros(len(sel), 3, device=device, requires_grad=True) if has_flare else None
        opt = torch.optim.Adam([x for x in (c, cf) if x is not None], lr=lr)
        for _ in range(steps):
            rec, kl, pred, rate = sde_terms(model, ex, c, cf)
            prior = (0.5 * ((c - mu_c) ** 2).sum() if c is not None else 0.0) + (0.5 * (cf ** 2).sum() if cf is not None else 0.0)
            loss = beta * (kl + prior / ex["valid"].sum()) + w_pred * pred + (rec + rate if cf is not None else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        out[idx] = torch.cat([x.detach() for x in (c, cf) if x is not None], 1).cpu().numpy()
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    return out
