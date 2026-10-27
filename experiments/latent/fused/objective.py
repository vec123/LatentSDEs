"""
The SDE terms of a batch (after model.encode): the score NLL at the visits from one posterior draw, the filtering KL per step, the
predictive score NLL from the SDE predict step of q_{t-1}, and with the flare part its terms (flare_terms); the multi-step rollout loss.
"""
import torch

from experiments.latent.fused.batch import ROLL_H, ROLL_STEP
from experiments.latent.hs_e2e import MONTH
from experiments.latent.parts.losses import gauss_kl, gauss_kl_logvar, laplace_nll
from experiments.latent.parts.masking import cut as stop_where


def score_nll(model, z, target, has, vt, off=None, cut=None):
    """off: an additive term of the mean in the normalized score space (the flare excess / ls); cut (B, N): rows where the slow part (g(z)
    with the visit-type effects) and the scale are fixed (no gradient): there only the flare excess learns"""
    mu, b = model.decode(z, vt)
    if cut is not None:
        mu, b = stop_where(mu, cut), stop_where(b, cut)
    if off is not None:
        mu = mu + off
    return laplace_nll(target, mu, b, has)


def flare_terms(model, ex, cf, sample=True):
    """the flare part of a batch after model.encode (see fused.flare.FlarePart.terms)"""
    m, lv = model.last_post
    return model.flare.terms(m, lv, model.last_h, ex, cf, sample)


def sde_terms(model, ex, c=None, cf=None):
    """score NLL at the visits from one posterior sample (reconstruction: q has seen the score), the filtering KL per step, and the
    predictive score NLL from one sample of the SDE predict step of q_{t-1} (the visit score before the visit is seen); c: the patient
    codes (B, d_code) of the SDE prior, None without codes; with flare (cf: the flare codes (B, 3)): the flare excess in both score NLLs,
    the amplitude KL in the KL, and the onset NLL (4th value, per step; 0 without flare)"""
    m, lv = model.last_post
    v, valid = lv.exp(), ex["valid"]
    B, N, _ = m.shape
    has, vt = ex["has"], ex["vt"]
    fl = flare_terms(model, ex, cf) if model.flare is not None else None
    off = (fl["F_post"] / ex["ls"], fl["F_pred"][:, 1:] / ex["ls"]) if fl else (None, None)
    # flare_cut (modular inference, a cut of the feedback): in the window after an onset the score NLLs train only the flare part, so that
    # z cannot lower the loss by following a flare (z and the amplitude explain the same spike; without the cut z took it, the diffusion
    # grew and the amplitude variance collapsed)
    cut = fl["win"] if fl and model.flare_cut > 0 else None
    if cut is not None:
        model.last_cut_share = float((cut * has).sum() / has.sum().clamp(min=1))
    rec = score_nll(model, m + (0.5 * lv).exp() * torch.randn_like(m), ex["sc"], has, vt, off[0], cut)
    cc = c[:, None, :].expand(B, N - 1, -1) if c is not None else None
    mp, vp = model.predict(m[:, :-1], v[:, :-1], ex["u"][:, 1:], ex["s"][:, None, :].expand(B, N - 1, -1), ex["dtm"][:, 1:, None], cc)
    vp = vp + 1e-4
    kl_t = gauss_kl(m[:, 1:], lv[:, 1:], mp, vp) * valid[:, 1:]
    kl_0 = gauss_kl_logvar(m[:, 0], lv[:, 0], model.z0_mean, model.z0_logvar)
    pred = score_nll(model, mp + vp.sqrt() * torch.randn_like(mp), ex["sc"][:, 1:], has[:, 1:], vt[:, 1:], off[1],
                     cut[:, 1:] if cut is not None else None)
    kl = kl_t.sum() + kl_0.sum() + (fl["kl_A"] if fl else 0.0)
    return rec, kl / valid.sum(), pred, (fl["rate_nll"] / valid.sum() if fl else torch.zeros((), device=m.device))


def rollout_loss(model, ex, rb, c=None, cf=None):
    """the multi-step rollout loss (trains the long-run behaviour of the drift and the set point): from the posterior mean at the anchor
    (no gradient into the posterior), the mean path of the SDE (no noise) on the ROLL_STEP grid with the own treatment path; flare models:
    + the excess of the flares up to the anchor (decaying, no gradient) and the expected burden of new flares; the Laplace NLL of the
    scheduled visits in the horizon (rb: fused.batch.roll_batch)"""
    m = model.last_post[0]
    B = m.shape[0]
    bi = torch.arange(B, device=m.device)
    z, s, h = m[bi, rb["anc"]].detach(), ex["s"], ROLL_STEP
    F = torch.zeros(B, device=m.device)
    fl = model.flare
    if fl is not None:
        with torch.no_grad():
            F = flare_terms(model, ex, cf, sample=False)["F_post"][bi, rb["anc"]]
        sc, sA = fl.effects(cf), fl.ls_A.exp()
    Z, Fs = [], []
    for k in range(ROLL_H):
        u = rb["u"][:, k]
        z = model.step(z, torch.zeros_like(z), u, s, h, c)[0]
        if fl is not None:
            pA, logD = fl.prior(u, s, sc)
            p_on = fl.onset_prob(z, u, s, sc, h * MONTH)
            F = F * torch.exp(-h * MONTH / logD.exp()) + p_on * torch.exp(pA + 0.5 * sA ** 2)
        Z.append(z)
        Fs.append(F)
    Z, Fs = torch.stack(Z, 1), torch.stack(Fs, 1)
    mu, b = model.decode(Z[bi[:, None], rb["tg"]], rb["tv"])
    mu = mu + Fs[bi[:, None], rb["tg"]] / ex["ls"]
    return laplace_nll(rb["ty"], mu, b, rb["tm"])
