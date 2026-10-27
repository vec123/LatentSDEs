"""
The data side of the SDE: per-row arrays (prepare), the extras of a batch padded as Data.pack (extras), and the anchors and targets of the
multi-step rollout loss (roll_batch).
"""
import numpy as np
import torch

from experiments.latent.hs_e2e import MONTH, ONEHOT

CONTROLS = ["Antibiotic_active", "Biologic_active", "bio_active_adalimumab", "bio_active_bimekizumab", "bio_active_secukinumab"]
STATIC = ["birth_year", "age_at_diagnosis", "bmi_at_baseline", "diagnostic_delay_years", "hurley_stage_baseline", "sex", "smoking_status",
          "family_history_hs"]
STEP_DAYS = 7.0
ROLL_H, ROLL_STEP, ROLL_J = 12, 1.0, 8  # the multi-step rollout loss: horizon (grid steps), step (months), max target visits per patient


def prepare(data):
    """the arrays of the SDE part: per row the gap to the previous step (months), the normalized log1p visit score and the treatment flags;
    per patient the static covariates (standardized value and mask flag)"""
    gap = np.diff(data.day, prepend=data.day[:1]).astype(np.float32) / MONTH
    gap[data.off[:-1][np.diff(data.off) > 0]] = 0
    data.dtm = np.maximum(gap, 0).astype(np.float32)
    lg = np.log1p(np.maximum(data.score, 0))
    if "lm" not in data.stats:
        tr = data.has_score & ~data.flag[data.prow]
        data.stats["lm"], data.stats["ls"] = float(lg[tr].mean()), float(lg[tr].std())
    data.sc = np.where(data.has_score, (lg - data.stats["lm"]) / data.stats["ls"], 0).astype(np.float32)
    nm = data.names.index
    data.u = np.stack([(data.raw[:, nm(c)] > 0) & data.M[:, nm(c)] for c in CONTROLS], 1).astype(np.float32)
    stat = STATIC  # with onehot: smoking status as its one-hot columns, and ethnicity (one-hot) also enters the prior
    if data.stats.get("onehot"):
        stat = [c for c in STATIC if c not in ONEHOT] + [n for n in data.names if n.split("=")[0] in ONEHOT and "=" in n]
    j, r0 = [nm(c) for c in stat], data.off[:-1]
    data.s = np.concatenate([data.X[r0][:, j], data.Mf[r0][:, j]], 1).astype(np.float32)
    data.s_names = list(stat) + [f"{c} missing" for c in stat]
    jv = nm("visit_type")  # codes of the train folder: 0 background, 1 flare_reactive, 2 follow_up, 3 treatment_monitoring; 4 no visit
    data.vt = np.clip(np.where(data.M[:, jv], data.raw[:, jv], 4), 0, 4).astype(np.int64)


def extras(data, pats, device, prior_rows=None):
    """the SDE inputs and targets of a batch, padded as data.pack; cp_in: the input of the code prior (the RNAseq sample seen at
    prior_rows, default the final step, x presence; presence; static covariates)"""
    lens = [data.n_rows(i) for i in pats]
    B, N = len(pats), max(lens)
    dt, sc, has, valid, day = (np.zeros((B, N), np.float32) for _ in range(5))
    u = np.zeros((B, N, len(CONTROLS)), np.float32)
    vt = np.full((B, N), 4, np.int64)
    for b, i in enumerate(pats):
        s, n = slice(data.off[i], data.off[i + 1]), lens[b]
        dt[b, :n], sc[b, :n], has[b, :n], u[b, :n], valid[b, :n] = data.dtm[s], data.sc[s], data.has_score[s], data.u[s], 1
        vt[b, :n], day[b, :n] = data.vt[s], data.day[s] - data.day[s][0]
    f = lambda a: torch.as_tensor(a, device=device)
    rows = np.asarray(prior_rows) if prior_rows is not None else np.array([data.off[i + 1] - 1 for i in pats])
    rr = data.rna_row[rows]
    pres = (rr >= 0).astype(np.float32)[:, None]
    cp_in = np.concatenate([data.rna_emb[np.clip(rr, 0, None)] * pres, pres, data.s[pats]], 1).astype(np.float32)
    return {"dtm": f(dt), "sc": f(sc), "has": f(has), "u": f(u), "s": f(data.s[pats]), "valid": f(valid), "vt": f(vt), "cp_in": f(cp_in),
            "day": f(day), "onset": f(((vt == 1) & (valid > 0)).astype(np.float32)), "ls": float(data.stats["ls"])}


def roll_batch(data, pats, rng, device):
    """anchors and targets of the multi-step rollout loss: per patient a random step t0 with >= 1 scored scheduled visit (background,
    treatment monitoring) in (t0, t0 + ROLL_H steps]; per grid step the treatment flags of the last row at or before its start (the
    patient's own path); the targets: up to ROLL_J such visits (grid index = the nearest step end, normalized score, visit type)"""
    B, K, J, H = len(pats), ROLL_H, ROLL_J, ROLL_H * ROLL_STEP * MONTH
    anc, u = np.zeros(B, np.int64), np.zeros((B, K, len(CONTROLS)), np.float32)
    tg, tv = np.zeros((B, J), np.int64), np.zeros((B, J), np.int64)
    ty, tm = np.zeros((B, J), np.float32), np.zeros((B, J), np.float32)
    for b, i in enumerate(pats):
        s = slice(data.off[i], data.off[i + 1])
        day = data.day[s]
        tgt = np.flatnonzero(data.has_score[s] & np.isin(data.vt[s], (0, 3)))
        if not len(tgt):
            continue
        dt = day[tgt]
        cand = np.flatnonzero(np.searchsorted(dt, day + H, side="right") > np.searchsorted(dt, day, side="right"))
        if not len(cand):
            continue
        k0 = int(rng.choice(cand))
        anc[b] = k0
        starts = day[k0] + ROLL_STEP * MONTH * np.arange(K)
        u[b] = data.u[data.off[i] + np.searchsorted(day, starts, side="right") - 1]
        sel = tgt[(dt > day[k0]) & (dt <= day[k0] + H)][:J]
        n = len(sel)
        tg[b, :n] = np.clip(np.round((day[sel] - day[k0]) / (ROLL_STEP * MONTH)).astype(int) - 1, 0, K - 1)
        ty[b, :n], tv[b, :n], tm[b, :n] = data.sc[s][sel], data.vt[s][sel], 1
    f = lambda a: torch.as_tensor(a, device=device)
    return {"anc": f(anc), "u": f(u), "tg": f(tg), "ty": f(ty), "tv": f(tv), "tm": f(tm)}
