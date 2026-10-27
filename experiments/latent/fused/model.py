"""
FusedSDE: the components together. At each step: the fused observation (encoders), the posterior (posterior), the forecast head on the
posterior and its skips (heads), the SDE prior (sde), the score decoder with the measurement model (decoders), optional view decoders,
patient codes (codes) and the flare part (flare). The names of the single-file model stay valid as properties (pcode, fl_rate, t_img,
last_att, ...), and checkpoints saved by it load (upgrade_state_dict maps the old keys).
"""
import torch
import torch.nn as nn

from experiments.latent.fused.batch import CONTROLS
from experiments.latent.fused.codes import PatientCodes
from experiments.latent.fused.decoders import ScoreDecoder, ViewDecoders
from experiments.latent.fused.encoders import FusionEncoder
from experiments.latent.fused.flare import FlarePart
from experiments.latent.fused.heads import LaplaceHead, head_input
from experiments.latent.fused.posterior import FilterPosterior
from experiments.latent.fused.sde import LatentSDE
from experiments.latent.parts.layers import gauss

# old key -> new key: exact names, then prefixes ("pcode_ls." before "pcode.")
_EXACT = {"null": "encoder.null", "type_img": "encoder.type_emb.img", "type_rna": "encoder.type_emb.rna", "vt_bias": "decoder.vt_bias",
          "vt_logs": "decoder.vt_logs", "lam_raw": "sde.lam_raw", "z0_mean": "sde.z0_mean", "z0_logvar": "sde.z0_logvar",
          "fl_ls_A": "flare.ls_A", "fl_ls_c": "flare.ls_c"}
_PREFIX = [("t_clin.", "encoder.query."), ("t_img.", "encoder.tok.img."), ("t_rna.", "encoder.tok.rna."), ("att.", "encoder.att."),
           ("norm.", "encoder.norm."), ("enc.", "encoder.out."), ("gru.", "posterior.gru."), ("post.", "posterior.out."),
           ("head.", "head.net."), ("dec_img.", "views.dec.img."), ("dec_rna.", "views.dec.rna."), ("dec.", "decoder.net."),
           ("set_point.", "sde.set_point."), ("drift.", "sde.drift."), ("diff.", "sde.diff."), ("w_level.", "sde.w_level."),
           ("w_noise.", "sde.w_noise."), ("w_speed.", "sde.w_speed."), ("w_resp.", "sde.w_resp."), ("pcode_ls.", "pcodes.log_sd."),
           ("pcode.", "pcodes.mean."), ("code_prior.", "pcodes.prior."), ("fl_rate.", "flare.rate."), ("fl_amp_p.", "flare.amp_prior."),
           ("fl_dur.", "flare.dur."), ("fl_amp_q.", "flare.amp_q."), ("pflare_ls.", "flare.codes_ls."), ("pflare.", "flare.codes.")]


def upgrade_state_dict(sd):
    """the keys of a checkpoint of the single-file model (hs_fused_sde.py before the split) mapped to the components; new keys unchanged"""
    if not any(k.startswith("gru.") for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        new = _EXACT.get(k)
        if new is None:
            new = next((n + k[len(o):] for o, n in _PREFIX if k.startswith(o)), k)
        out[new] = v
    return out


class FusedSDE(nn.Module):
    def __init__(self, n_feat, d_img, d_rna, d_u, d_s, kind_cols, dz=16, hidden=128, d_tok=64, heads=4, meas=False, recon=False,
                 recon_latent=False, head_obs=False, head_gru=False, drift_mr=False, sde_sub=0.0, n_codes=0, d_code=0, code_struct=(),
                 code_var=False, code_prior_rna=False, flare=False, flare_cut=0.0):
        super().__init__()
        self.dz, self.kind_cols, self.meas, self.head_obs, self.head_gru = dz, list(kind_cols), meas, head_obs, head_gru
        self.code_struct, self.drift_mr, self.sde_sub = tuple(code_struct), drift_mr, sde_sub
        self.d_code = sum(self.code_struct) if self.code_struct else d_code
        if self.d_code:
            assert drift_mr, "the patient code needs the mean-reverting drift (--drift_mr 1)"
        if code_prior_rna:
            assert self.code_struct, "the RNAseq prior needs the structured code (--code_struct)"
        views = {"img": d_img, "rna": d_rna}
        self.encoder = FusionEncoder(2 * n_feat, views, d_tok, heads, dz)
        self.posterior = FilterPosterior(dz, hidden)
        self.head = LaplaceHead((4 if head_obs else 2) * dz + (hidden if head_gru else 0) + len(self.kind_cols))
        self.sde = LatentSDE(dz, d_u, d_s, self.d_code, self.code_struct, drift_mr, sde_sub)
        self.decoder = ScoreDecoder(dz, meas)
        self.views = ViewDecoders(dz if recon_latent else d_tok, views) if recon else None
        self.pcodes = (PatientCodes(n_codes, self.d_code, code_var, d_rna + 1 + d_s, self.code_struct[3] if code_prior_rna else 0)
                       if self.d_code else None)
        self.flare = FlarePart(dz, d_s, hidden, n_codes, flare_cut) if flare else None
        self.last_obs = self.last_post = self.last_h = None
        self.last_cut_share = 0.0

    def load_state_dict(self, state_dict, strict=True, **kw):
        return super().load_state_dict(upgrade_state_dict(state_dict), strict=strict, **kw)

    # the steps of the model
    def fuse(self, b):
        """the fused observation (mean, logvar) of every step"""
        q = torch.cat([b["x"], b["m"]], -1)
        return self.encoder(q, {"img": (b["img"], b["img_p"]), "rna": (b["rna"], b["rna_p"])})

    def encode(self, b):
        """the posterior at every step; returns the input of the forecast head (kept: last_obs, last_post, last_h)"""
        mo, lo = self.fuse(b)
        self.last_obs = (mo, lo)
        m, lv, h = self.posterior(mo, lo, b["dt"])
        self.last_post, self.last_h = (m, lv), h
        return head_input(m, lv, b["x"][..., self.kind_cols], (mo, lo) if self.head_obs else None, h if self.head_gru else None)

    def forecast(self, h):
        return self.head(h)

    def decode(self, z, vt=None):
        return self.decoder(z, vt)

    def step(self, m, v, u, s, h, c=None):
        return self.sde.step(m, v, u, s, h, c)

    def predict(self, m, v, u, s, dt, c=None):
        return self.sde.predict(m, v, u, s, dt, c)

    def lam(self, c_speed=None):
        return self.sde.lam(c_speed)

    def fg(self, z, u, s):
        return self.sde.fg(z, u, s)

    def split_code(self, c, u):
        return self.sde.split_code(c, u)

    def code_prior_mean(self, x):
        return self.pcodes.prior_mean(x) if self.pcodes is not None else torch.zeros(x.shape[0], 0, device=x.device)

    gauss = staticmethod(gauss)

    # the names of the single-file model
    codes = property(lambda self: self.encoder.tokens)  # the view tokens of the last batch (image, RNAseq)
    last_att = property(lambda self: self.encoder.last_att)
    t_clin = property(lambda self: self.encoder.query)
    t_img = property(lambda self: self.encoder.tok["img"])
    t_rna = property(lambda self: self.encoder.tok["rna"])
    vt_bias = property(lambda self: self.decoder.vt_bias)
    vt_logs = property(lambda self: self.decoder.vt_logs)
    dec_img = property(lambda self: self.views.dec["img"])
    dec_rna = property(lambda self: self.views.dec["rna"])
    lam_raw = property(lambda self: self.sde.lam_raw)
    z0_mean = property(lambda self: self.sde.z0_mean)
    z0_logvar = property(lambda self: self.sde.z0_logvar)
    pcode = property(lambda self: self.pcodes.mean)
    pcode_ls = property(lambda self: self.pcodes.log_sd)
    code_var = property(lambda self: bool(self.pcodes is not None and self.pcodes.var))
    flare_cut = property(lambda self: self.flare.cut if self.flare is not None else 0.0)
    fl_rate = property(lambda self: self.flare.rate)
    fl_amp_p = property(lambda self: self.flare.amp_prior)
    fl_dur = property(lambda self: self.flare.dur)
    fl_amp_q = property(lambda self: self.flare.amp_q)
    fl_ls_A = property(lambda self: self.flare.ls_A)
    fl_ls_c = property(lambda self: self.flare.ls_c)
    pflare = property(lambda self: self.flare.codes)
    pflare_ls = property(lambda self: self.flare.codes_ls)


def make_model(data, a):
    """the model of a run's arguments (a run without the newer keys gets their defaults)"""
    return FusedSDE(data.X.shape[1], data.img_emb.shape[1], data.rna_emb.shape[1], len(CONTROLS), data.s.shape[1],
                    [data.names.index("is_query"), data.names.index("is_final")], dz=a.get("dz", 16), hidden=a.get("hidden", 128),
                    meas=bool(a.get("meas", 0)), recon=a.get("w_recon", 0) > 0, recon_latent=bool(a.get("recon_latent", 0)),
                    head_obs=bool(a.get("head_obs", 0)), head_gru=bool(a.get("head_gru", 0)), drift_mr=bool(a.get("drift_mr", 0)),
                    sde_sub=float(a.get("sde_sub", 0)), n_codes=data.n_pat, d_code=int(a.get("code_dim", 0)),
                    code_struct=tuple(int(k) for k in str(a.get("code_struct", "")).split(",") if k),
                    code_var=bool(a.get("code_var", 0)), code_prior_rna=bool(a.get("code_prior_rna", 0)), flare=bool(a.get("flare", 0)),
                    flare_cut=float(a.get("flare_cut", 0)))
