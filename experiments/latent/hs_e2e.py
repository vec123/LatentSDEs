"""
End-to-end model for the three tasks of the challenge: one forward GRU over the step sequences of hs_build.py, with two heads.

Inputs of a step: the features (standardized, 0 where missing) and their mask flags, the log gap to the previous step, and the image
and RNAseq embeddings from their sample date on, with a presence flag (16 principal components of each, fitted on the train folder,
projected to d_mod values; the first run read the full width). In training each modality of a patient is hidden with probability
p_drop, so that one model runs with and without it (task 3), and the embeddings get a Gaussian noise (against memorizing a patient by
a unique vector).
Head 1 (tasks 1 to 3): the mean IHS4 of the 6 months after the step, as a median (L1) and a Laplace scale. Labels: the derived ones
at visits and quarter ends (each patient weight 1/sqrt(n) per kind), the real one at the final step (weight 1). The step kind is an
input, so the model learns the offset of the real label (1.4 lower than the visit-based labels of the same patients).
Head 2 (task 2): the mean score of the 6 months after a horizon D, given a plan: no biologic, or drug d active through the window
after tau months on it. Trained on random (step, horizon) pairs of the training patients, with the plan read from their treatment
record (pairs with other treatment paths are skipped). From the decision step of a start, plan "none" is the counterfactual
without treatment. v2: head 2 also reads the modality features of its step, windows that end after the data count with their
observed part (the observed share is an input; a prediction asks for the share 1), and the pairs from a decision step with an RNAseq
sample have the weight rna_weight (most RNAseq starts are in 2025, so their windows are rarely complete).
Causal: the GRU runs forward, so the state at a step reads the steps up to it (checked on held-out patients).
Training stops when the MAE on the real labels of the validation patients (10% of the training patients) stops improving.

    python -m experiments.latent.hs_e2e [--seconds 600] [--smoke]
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import yaml

from experiments.latent.parts.losses import laplace_terms, view_recon as recon_terms  # noqa: F401  (shared losses; the old names)

PATHS = {f: f"test_data/hs_public_dataset/hs_e2e_{f}.npz" for f in ("train", "test_1", "test_2")}
VISIT, QUERY, FINAL, DECISION = 0, 1, 2, 3
MONTH = 30.44
DRUGS = ["Adalimumab", "Bimekizumab", "Secukinumab"]
HIDE = ["is_decision"]  # read by the analysis only: a decision step must look like any step without an observation
# raw input (--raw 1): the history summaries of hs_build.py are not read, so the GRU must build the history from the steps. Every scored
# visit is a step and a quarter-end step comes at least every 3 months, so no score is lost. Kept: the observation of the step (score, visit
# type, month), the step kind, the clocks (time, age, time since diagnosis / onset), the static features and the current treatments.
RAW_DROP = ["mean_3m", "n_3m", "std_3m", "mean_6m", "n_6m", "std_6m", "mean_12m", "n_12m", "std_12m", "mean_prev6m", "n_prev6m", "trend_6m",
            "mean_all", "n_scores_all", "last_score", "days_since_score", "n_visits_so_far", "days_since_prev_visit", "vis6m_background",
            "vis6m_flare_reactive", "vis6m_follow_up", "vis6m_treatment_monitoring", "days_since_tx_start", "max_bio_line", "days_on_bio",
            "surg_recur_rate", "days_since_surg", "last_wound_weeks", "n_surg_1y", "img_days_since", "rna_days_since"]
# categorical features with more than 2 values (hs_build.py codes them as one ordinal number, alphabetical order): with onehot, one static
# column per code (number of codes of the train folder) and the ordinal column hidden from the input (data.raw keeps it for the analysis)
ONEHOT = {"ethnicity": 5, "smoking_status": 3}
# event counts from the record start: with --raw 1 read as the number of events since the previous step (the first step: since the record start)
RAW_INC = ["n_tx_started", "n_tx_ended", "Antibiotic_ever", "Biologic_ever", "n_surg", "surg_CO2 laser ablation", "surg_Deroofing",
           "surg_Incision & drainage", "surg_Wide local excision"]


def held_out(flag, split, fold=None):
    """The held-out patients of the train folder: split 0 the is_test_patient flag of hs_build.py; split k > 0 a new set of the same
    size, drawn (seed k) from the other patients, so that it has no patient of the first set. fold (f, K): cross-validation fold f of K, where
    fold 0 is the is_test_patient set and folds 1..K-1 split the other patients at random (seed 0). xgb_e2e.py has the same function."""
    if flag.all():
        return flag
    if fold is not None:
        f, k = fold
        if f == 0:
            return flag
        new = np.zeros_like(flag)
        new[np.array_split(np.random.default_rng(0).permutation(np.flatnonzero(~flag)), k - 1)[f - 1]] = True
        return new
    if split == 0:
        return flag
    new = np.zeros_like(flag)
    new[np.random.default_rng(split).choice(np.flatnonzero(~flag), int(flag.sum()), replace=False)] = True
    return new


class Data:
    """The steps of one folder as standardized arrays. stats: the transforms of the train folder (None: compute them here, with
    emb_pca > 0 principal components of each embedding instead of the full width, the held-out set `split` and the features `drop`
    removed from the input, raw: no history summaries in the input (RAW_DROP, RAW_INC); the stats keep the three)."""

    def __init__(self, path, stats=None, emb_pca=0, split=0, drop=(), fold=None, raw=False, mod_events=False, onehot=False):
        d = np.load(path)
        self.names = [str(n) for n in d["feature_names"]]
        X, self.M = d["X"].astype(np.float32), d["M"] > 0
        static = d["static_feature"] > 0
        if stats is not None:
            onehot = stats.get("onehot", False)
        if onehot:
            for c, k in ONEHOT.items():
                j = self.names.index(c)
                X = np.concatenate([X, np.stack([(X[:, j] == v) & self.M[:, j] for v in range(k)], 1).astype(np.float32)], 1)
                self.M = np.concatenate([self.M, np.repeat(self.M[:, j:j + 1], k, 1)], 1)
                static = np.concatenate([static, np.ones(k, bool)])
                self.names += [f"{c}={v}" for v in range(k)]
        self.kind, self.day = d["kind"].astype(np.int64), d["day"].astype(np.int64)
        self.y, self.ym = d["y"].astype(np.float32), d["y_mask"] > 0
        self.off, self.pid = d["offsets"], d["patient_id"]
        if stats is not None:
            split, drop, fold = stats.get("split", 0), stats.get("drop", []), stats.get("fold")
        self.flag = held_out(d["is_test_patient"] > 0, split, fold)
        self.n_pat = len(self.pid)
        self.raw = X.copy()  # features before the transforms (the analysis reads them)
        self.prow = np.repeat(np.arange(self.n_pat), np.diff(self.off))
        own = self.names.index("own_score")
        self.has_score, self.score = (self.kind == VISIT) & self.M[:, own], X[:, own]
        self.end_day = int(self.day[self.kind == FINAL].max())
        if stats is None:
            train = ~self.flag[self.prow]
            logc = [j for j, nme in enumerate(self.names) if not static[j] and nme != "time_months" and self.M[train, j].any()
                    and X[train & self.M[:, j], j].min() >= 0 and X[train & self.M[:, j], j].max() > 50]
            Xl = X.copy()
            Xl[:, logc] = np.log1p(np.maximum(Xl[:, logc], 0))
            fit = train[:, None] & self.M
            cnt = fit.sum(0).clip(min=1)
            mean = np.where(fit, Xl, 0).sum(0) / cnt
            std = np.sqrt(np.where(fit, (Xl - mean) ** 2, 0).sum(0) / cnt)
            std[std < 1e-6] = 1.0
            yt = self.y[train & self.ym]
            stats = {"logc": logc, "mean": mean, "std": std, "ymean": float(yt.mean()), "ystd": float(yt.std()),
                     "img": (d["img_emb"].mean(0), d["img_emb"].std(0) + 1e-6), "rna": (d["rna_emb"].mean(0), d["rna_emb"].std(0) + 1e-6),
                     "split": int(split), "drop": list(drop), "fold": list(fold) if fold is not None else None, "raw": bool(raw),
                     "mod_events": bool(mod_events), "onehot": bool(onehot)}
            if raw:
                inc = self.increments()[train]
                stats["inc"] = (inc.mean(0), inc.std(0) + 1e-6)
            if emb_pca:  # principal components of the embeddings of the train folder (no label is used), each scaled to unit variance
                for k in ("img", "rna"):
                    E = d[f"{k}_emb"].astype(np.float64)
                    mu = E.mean(0)
                    comps = np.linalg.svd(E - mu, full_matrices=False)[2][:emb_pca]
                    z = (E - mu) @ comps.T
                    stats[f"{k}_pca"] = (mu, comps, z.std(0) + 1e-6)
        self.stats = stats
        X[:, stats["logc"]] = np.log1p(np.maximum(X[:, stats["logc"]], 0))
        self.X = np.where(self.M, (X - stats["mean"]) / stats["std"], 0).astype(np.float32)
        self.Mf = self.M.astype(np.float32)
        for c in HIDE + list(stats.get("drop", [])) + (RAW_DROP if stats.get("raw") else []) + (list(ONEHOT) if stats.get("onehot") else []):
            j = self.names.index(c)
            self.X[:, j], self.Mf[:, j] = 0, 0
        if stats.get("raw"):
            j = [self.names.index(c) for c in RAW_INC]
            self.X[:, j] = np.where(self.M[:, j], (self.increments() - stats["inc"][0]) / stats["inc"][1], 0).astype(np.float32)
        self.ys = ((self.y - stats["ymean"]) / stats["ystd"]).astype(np.float32)
        for k in ("img", "rna"):
            E = d[f"{k}_emb"]
            if f"{k}_pca" in stats:
                mu, comps, zs = stats[f"{k}_pca"]
                E = ((E - mu) @ comps.T) / zs
            else:
                E = (E - stats[k][0]) / stats[k][1]
            setattr(self, f"{k}_emb", E.astype(np.float32))
        self.real_plan = False  # v3: also head 2 examples on the real label
        self.partial, self.rna_weight = False, 1.0  # head 2 examples: windows that end after the data; weight of RNAseq decision steps
        self.img_row, self.rna_row = d["img_row"].astype(np.int64), d["rna_row"].astype(np.int64)
        # the samples in the input: from the sample date on (img_row, rna_row), or with mod_events only at the first step that sees a sample
        self.img_in, self.rna_in = self.img_row, self.rna_row
        if stats.get("mod_events"):
            self.img_in, self.rna_in = self.first_seen(self.img_row), self.first_seen(self.rna_row)
        self.img_new, self.rna_new = self.first_seen(self.img_row) >= 0, self.first_seen(self.rna_row) >= 0  # the steps of a new sample
        dt = np.diff(self.day, prepend=self.day[:1]).astype(np.float32)
        dt[self.off[:-1][np.diff(self.off) > 0]] = 0
        self.dt = (np.log1p(np.maximum(dt, 0)) / 5.0).astype(np.float32)
        # label weights of head 1: per patient 1/sqrt(n) for each kind of derived label, 1 for the real label
        self.w = np.zeros(len(self.y), np.float32)
        for kd in (VISIT, QUERY):
            lab = self.ym & (self.kind == kd)
            cnt = np.bincount(self.prow[lab], minlength=self.n_pat)
            self.w[lab] = 1.0 / np.sqrt(cnt[self.prow[lab]])
        self.w[self.ym & (self.kind == FINAL)] = 1.0
        # biologic intervals per patient: (start day, end day, drug)
        bp = d["bio_patient"]
        o = np.argsort(bp, kind="stable")
        cuts = np.searchsorted(bp[o], np.arange(self.n_pat + 1))
        bs, be, bd = d["bio_start"][o].astype(np.int64), d["bio_end"][o].astype(np.int64), d["bio_drug"][o].astype(np.int64)
        self.bio = [(bs[cuts[i]:cuts[i + 1]], be[cuts[i]:cuts[i + 1]], bd[cuts[i]:cuts[i + 1]]) for i in range(self.n_pat)]

    def n_rows(self, i):
        return int(self.off[i + 1] - self.off[i])

    def final_row(self, i):
        s = slice(self.off[i], self.off[i + 1])
        return self.off[i] + int(np.flatnonzero(self.kind[s] == FINAL)[0])

    def plan_examples(self, i, rng, k, partial=None):
        """k candidate (step, horizon) pairs of patient i; the valid ones as (local step, horizon months, tau months, plan, target,
        weight, observed share of the window). partial: a window may end after the data (the mean of its observed part, at least 30
        days, is the target, and the observed share is an input of head 2; a prediction asks for the share 1)."""
        partial = self.partial if partial is None else partial
        s0, s1 = self.off[i], self.off[i + 1]
        kinds, days = self.kind[s0:s1], self.day[s0:s1]
        cand, dec = np.flatnonzero(kinds != FINAL), np.flatnonzero(kinds == DECISION)
        sc = self.has_score[s0:s1]
        sd, sv = days[sc], self.score[s0:s1][sc]
        if len(cand) == 0 or len(sd) == 0:
            return []
        st, en, dr = self.bio[i]
        out = []
        for _ in range(k):
            kk = int(rng.choice(dec)) if len(dec) and rng.random() < 0.5 else int(rng.choice(cand))
            t = days[kk]
            hz = rng.uniform(0, 12) if rng.random() < 0.5 else float(np.exp(rng.uniform(np.log(12), np.log(108))))
            ws, we = t + hz * MONTH, t + (hz + 6) * MONTH
            we_obs = min(we, self.end_day)
            if (we > self.end_day and not partial) or we_obs - ws < 30:
                continue
            inw = (sd > ws) & (sd <= we_obs)
            if not inw.any():
                continue
            ov = (st <= we_obs) & (en >= t)
            if not ov.any():
                plan, tau = 0, 0.0
            elif ov.sum() == 1 and st[ov][0] <= ws and en[ov][0] >= we_obs:
                plan, tau = int(dr[ov][0]) + 1, (ws - st[ov][0]) / MONTH
            else:
                continue
            wgt = self.rna_weight if kinds[kk] == DECISION and self.rna_row[s0 + kk] >= 0 else 1.0
            out.append((kk, hz, tau, plan, float(sv[inw].mean()), wgt, (we_obs - ws) / (we - ws), 0.0))
        if self.real_plan:
            out += self.real_examples(i, rng)
        return out

    def real_examples(self, i, rng):
        """head 2 examples on the real label (flag 1): from the decision step of the open line at the cutoff, with its drug; for a
        patient who never had a biologic, from a random step with plan none. Horizon and tau as in hs_task2.counterfactual."""
        fr = self.final_row(i)
        if not self.ym[fr]:
            return []
        s0, s1 = self.off[i], self.off[i + 1]
        st, en, dr = self.bio[i]
        cut, y = self.end_day, float(self.y[fr])
        act = (st <= cut) & (en >= cut)
        if act.any():
            j = np.flatnonzero(act)[np.argmax(st[act])]
            dec = np.flatnonzero((self.kind[s0:s1] == DECISION) & (self.day[s0:s1] == st[j] - 1))
            if not len(dec):
                return []
            kk = int(dec[0])
            wgt = self.rna_weight if self.rna_row[s0 + kk] >= 0 else 1.0
            return [(kk, (cut - (st[j] - 1)) / MONTH, (cut - st[j]) / MONTH, int(dr[j]) + 1, y, wgt, 1.0, 1.0)]
        if len(st) == 0:
            cand = np.flatnonzero(self.kind[s0:s1] != FINAL)
            if len(cand):
                kk = int(rng.choice(cand))
                return [(kk, (cut - self.day[s0 + kk]) / MONTH, 0.0, 0, y, 1.0, 1.0, 1.0)]
        return []

    def increments(self):
        """the RAW_INC counts as the number of events since the previous step of the patient (the first step: the count itself)"""
        j = [self.names.index(c) for c in RAW_INC]
        v = np.where(self.M[:, j], self.raw[:, j], 0)
        d = np.diff(v, axis=0, prepend=0)
        first = self.off[:-1][np.diff(self.off) > 0]
        d[first] = v[first]
        return d.astype(np.float32)

    def first_seen(self, row):
        """row (the latest sample on or before each step, -1 none) kept only at the first step of the patient that sees the sample"""
        new = (row >= 0) & (row != np.concatenate([[-1], row[:-1]]))
        first = self.off[:-1][np.diff(self.off) > 0]
        new[first] = row[first] >= 0
        return np.where(new, row, -1)

    def pack(self, pats, device, drop=0.0, rng=None, plan_per=0, hide_img=False, hide_rna=False, hide_cols=None, noise=0.0):
        """hide_cols: feature columns shown as missing at every step (the driver analysis); noise: the standard deviation of a Gaussian
        noise on the embeddings (training)"""
        B, lens = len(pats), [self.n_rows(i) for i in pats]
        N, F = max(lens), self.X.shape[1]
        x, m = np.zeros((B, N, F), np.float32), np.zeros((B, N, F), np.float32)
        dt, y, w = np.zeros((B, N, 1), np.float32), np.zeros((B, N), np.float32), np.zeros((B, N), np.float32)
        ir, rr = np.full((B, N), -1, np.int64), np.full((B, N), -1, np.int64)
        ni, nr = np.zeros((B, N), np.float32), np.zeros((B, N), np.float32)  # 1 at the step of a new (shown) sample: reconstruction targets
        ti, tr = np.full((B, N), -1, np.int64), np.full((B, N), -1, np.int64)  # the new samples, shown or hidden (targets of the latent)
        plans = []
        for b, i in enumerate(pats):
            s, n = slice(self.off[i], self.off[i + 1]), lens[b]
            x[b, :n], m[b, :n], dt[b, :n, 0] = self.X[s], self.Mf[s], self.dt[s]
            ti[b, :n] = np.where(self.img_new[s], self.img_row[s], -1)
            tr[b, :n] = np.where(self.rna_new[s], self.rna_row[s], -1)
            if hide_cols is not None:
                x[b, :n][:, hide_cols], m[b, :n][:, hide_cols] = 0, 0
            y[b, :n], w[b, :n] = self.ys[s], self.w[s]
            if not (hide_img or (drop and rng.random() < drop)):
                ir[b, :n], ni[b, :n] = self.img_in[s], self.img_new[s]
            if not (hide_rna or (drop and rng.random() < drop)):
                rr[b, :n], nr[b, :n] = self.rna_in[s], self.rna_new[s]
            if plan_per:
                plans += [(b,) + e for e in self.plan_examples(i, rng, plan_per)]
        f = lambda a: torch.as_tensor(a, device=device)
        img, rna = self.img_emb[np.clip(ir, 0, None)], self.rna_emb[np.clip(rr, 0, None)]
        img_t, rna_t = self.img_emb[np.clip(ti, 0, None)], self.rna_emb[np.clip(tr, 0, None)]  # without the noise: reconstruction targets
        if noise:
            img = img + noise * rng.standard_normal(img.shape).astype(np.float32)
            rna = rna + noise * rng.standard_normal(rna.shape).astype(np.float32)
        out = {"x": f(x), "m": f(m), "dt": f(dt), "y": f(y), "w": f(w), "len": lens,
               "img": f(img * (ir >= 0)[..., None]), "img_p": f((ir >= 0)[..., None].astype(np.float32)),
               "rna": f(rna * (rr >= 0)[..., None]), "rna_p": f((rr >= 0)[..., None].astype(np.float32)),
               "img_t": f(img_t), "img_n": f(ni), "rna_t": f(rna_t), "rna_n": f(nr),
               "img_a": f((ti >= 0).astype(np.float32)), "rna_a": f((tr >= 0).astype(np.float32))}
        if plans:
            P = np.array(plans, dtype=np.float64)
            out["plan"] = {"b": f(P[:, 0].astype(np.int64)), "k": f(P[:, 1].astype(np.int64)), "hz": f(P[:, 2].astype(np.float32)),
                           "tau": f(P[:, 3].astype(np.float32)), "plan": f(P[:, 4].astype(np.int64)),
                           "t": f(((P[:, 5] - self.stats["ymean"]) / self.stats["ystd"]).astype(np.float32)),
                           "w": f(P[:, 6].astype(np.float32)), "obs": f(P[:, 7].astype(np.float32)),
                           "real": f(P[:, 8].astype(np.float32))}
        return out


class E2E(nn.Module):
    """v2: head 2 also reads the projected modality features of its step (so that a baseline sample can change the response
    directly) and the observed share of its window"""

    def __init__(self, n_feat, d_img, d_rna, d_mod=16, hidden=128, dropout=0.1, v2=False, v3=False, recon=False, recon_latent=False):
        super().__init__()
        self.n_plan, self.v2, self.v3 = 1 + len(DRUGS), v2, v3
        self.img, self.rna = nn.Linear(d_img, d_mod), nn.Linear(d_rna, d_mod)
        if recon:  # ALDSL view term: each code (or with recon_latent the GRU state, shared by the views) rebuilds the full embedding
            d_src = hidden if recon_latent else d_mod
            self.dec_img, self.dec_rna = nn.Linear(d_src, d_img), nn.Linear(d_src, d_rna)
        self.codes = None
        self.inp = nn.Sequential(nn.Linear(2 * n_feat + 1 + 2 * (d_mod + 1), hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))
        d_plan = hidden + 4 + self.n_plan + (2 * (d_mod + 1) + 1 if v2 else 0) + (1 if v3 else 0)
        self.plan_head = nn.Sequential(nn.Linear(d_plan, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        self.last_mod = None

    def encode(self, b):
        ei = torch.tanh(self.img(b["img"])) * b["img_p"]
        er = torch.tanh(self.rna(b["rna"])) * b["rna_p"]
        self.codes = (ei, er)
        self.last_mod = torch.cat([ei, b["img_p"], er, b["rna_p"]], -1)  # (B, N, 2 (d_mod + 1)): read by plan_at
        u = torch.cat([b["x"], b["m"], b["dt"], self.last_mod], -1)
        return self.gru(self.inp(u))[0]

    def forecast(self, h):
        o = self.head(h)
        return o[..., 0], Fn.softplus(o[..., 1]) + 1e-3

    def plan_forecast(self, hk, hz, tau, plan, mod=None, obs=None, real=None):
        """real (v3): 1 if the target is on the scale of the real label, 0 (default) for visit-based windows"""
        enc = lambda v: torch.stack([v / 60.0, torch.log1p(v) / 4.0], -1)
        z = [hk, enc(hz), enc(tau), Fn.one_hot(plan, self.n_plan).float()]
        if self.v2:
            z += [mod, (torch.ones_like(hz) if obs is None else obs)[:, None]]
        if self.v3:
            z += [(torch.zeros_like(hz) if real is None else real)[:, None]]
        o = self.plan_head(torch.cat(z, -1))
        return o[..., 0], Fn.softplus(o[..., 1]) + 1e-3


def plan_at(model, h, bi, ki, hz, tau, plan, obs=None, real=None):
    """head 2 at the steps (bi, ki) of the last encoded batch; obs None: the full window; real: label-type flag (v3)"""
    return model.plan_forecast(h[bi, ki], hz, tau, plan, model.last_mod[bi, ki] if model.v2 else None, obs, real)


def make_model(data, a):
    """the model of a run's arguments (a run before v2 lacks the newer keys: their defaults give its model)"""
    return E2E(data.X.shape[1], data.img_emb.shape[1], data.rna_emb.shape[1], d_mod=a.get("d_mod", 16), hidden=a.get("hidden", 128),
               v2=bool(a.get("v2", 0)), v3=bool(a.get("v3", 0)), recon=a.get("w_recon", 0) > 0, recon_latent=bool(a.get("recon_latent", 0)))


@torch.no_grad()
def predict(model, data, pats, device, rows="final", hide_img=False, hide_rna=False, chunk=50, pause=0.0):
    """head 1 at the final step of each patient (rows='final': arrays per patient), or at every step (rows='all': per row)"""
    model.eval()
    ys, ym = data.stats["ystd"], data.stats["ymean"]
    order = np.argsort([data.n_rows(i) for i in pats])
    mu_out, b_out = np.zeros(len(pats)), np.zeros(len(pats))
    all_mu = np.full(len(data.y), np.nan, np.float32) if rows == "all" else None
    for a in range(0, len(order), chunk):
        sel = [pats[j] for j in order[a:a + chunk]]
        batch = data.pack(sel, device, hide_img=hide_img, hide_rna=hide_rna)
        mu, b = model.forecast(model.encode(batch))
        mu, b = (mu * ys + ym).cpu().numpy(), (b * ys).cpu().numpy()
        for j, i in zip(order[a:a + chunk], sel):
            n = data.n_rows(i)
            k = data.final_row(i) - data.off[i]
            mu_out[j], b_out[j] = mu[sel.index(i), k], b[sel.index(i), k]
            if all_mu is not None:
                all_mu[data.off[i]:data.off[i] + n] = mu[sel.index(i), :n]
        time.sleep(pause)
    model.train()
    return (mu_out, b_out) if rows == "final" else all_mu


def train(model, data, args, device, run_dir, log):
    rng = np.random.default_rng(args.seed)
    tr, val = data.train_idx, data.val_idx
    lengths = np.array([data.n_rows(i) for i in tr])
    yv = np.array([data.y[data.final_row(i)] for i in val])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    start, step, best, best_step, best_state, stop = time.time(), 0, float("inf"), 0, None, False
    while step < args.steps and time.time() - start < args.seconds and not stop:
        order = np.argsort(lengths + rng.uniform(0, 3, len(lengths)))
        parts = [order[a:a + args.batch] for a in range(0, len(order), args.batch)]
        for p in rng.permutation(len(parts)):
            if step >= args.steps or time.time() - start >= args.seconds:
                break
            batch = data.pack([int(tr[k]) for k in parts[p]], device, drop=args.p_drop, rng=rng, plan_per=args.plan_per, noise=args.emb_noise)
            h = model.encode(batch)
            mu, b = model.forecast(h)
            l1, nll = laplace_terms(batch["y"], mu, b, batch["w"])
            loss, l2 = l1 + 0.1 * nll, torch.zeros(())
            if "plan" in batch:
                P = batch["plan"]
                mu2, b2 = plan_at(model, h, P["b"], P["k"], P["hz"], P["tau"], P["plan"], P["obs"], P["real"])
                l2, nll2 = laplace_terms(P["t"], mu2, b2, P["w"])
                loss = loss + args.plan_weight * (l2 + 0.1 * nll2)
            ri = rr = torch.zeros(())
            if args.w_recon > 0:
                ri, rr = recon_terms((h, h) if args.recon_latent else model.codes, (model.dec_img, model.dec_rna), batch,
                                     hidden_too=bool(args.recon_latent))
                loss = loss + args.w_recon * (ri + rr)
            opt.zero_grad()
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)):
                opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                msg = f"{step} t={time.time() - start:.0f}s head1 L1 {l1.item():.4f} head2 L1 {l2.item():.4f}" + (
                    f" reconstruction MSE (image, RNAseq) {ri.item():.3f} {rr.item():.3f}" if args.w_recon > 0 else "")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.train_pause:
                time.sleep(args.train_pause)
            if step % args.val_every == 0:
                v = float(np.mean(np.abs(predict(model, data, list(val), device)[0] - yv)))
                if v < best:
                    best, best_step, best_state = v, step, {k: x.detach().clone() for k, x in model.state_dict().items()}
                msg = f"{step} validation MAE (real labels) {v:.4f} (best {best:.4f} at {best_step})"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                if step - best_step >= args.patience * args.val_every:
                    print(f"no improvement for {args.patience} checks: stop", flush=True)
                    stop = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"model of step {best_step} (validation MAE {best:.4f}) is kept", flush=True)
    return step, best_step, best


def evaluate(model, data, device, run_dir, pause=0.0, max_patients=None):
    """Tasks 1 and 3 (head 1) on the held-out patients of the train folder; writes metrics.yaml and final_preds.npz."""
    te = list(data.test_idx[:max_patients]) if max_patients else list(data.test_idx)
    yt = np.array([data.y[data.final_row(i)] for i in te])
    fr = np.array([data.final_row(i) for i in te])
    nm = data.names.index
    var = {"full": predict(model, data, te, device, pause=pause),
           "no_img": predict(model, data, te, device, hide_img=True, pause=pause),
           "no_rna": predict(model, data, te, device, hide_rna=True, pause=pause),
           "clinical": predict(model, data, te, device, hide_img=True, hide_rna=True, pause=pause)}
    has_img, has_rna = data.img_row[fr] >= 0, data.rna_row[fr] >= 0
    no_visits = data.raw[fr, nm("n_visits_so_far")] == 0
    gap = np.where(data.M[fr, nm("days_since_prev_visit")], data.raw[fr, nm("days_since_prev_visit")], np.inf)
    bio = data.raw[fr, nm("Biologic_active")] > 0
    groups = {"all held-out patients": np.ones(len(te), bool), "without visits": no_visits, "last visit > 90 days before": gap > 90,
              "open biologic line (task 2)": bio, "with image (task 3a)": has_img, "with image and RNAseq (task 3b)": has_img & has_rna}
    out = {}
    print(f"\nheld-out patients: MAE of head 1 at the final step (real labels)")
    print(f"{'group':34s} {'n':>5s} {'full':>7s} {'no image':>9s} {'no RNA':>7s} {'clinical':>9s}")
    for g, s in groups.items():
        if s.sum() == 0:
            continue
        r = {k: float(np.mean(np.abs(v[0][s] - yt[s]))) for k, v in var.items()}
        out[g] = {"n": int(s.sum()), **r}
        print(f"{g:34s} {s.sum():5d} {r['full']:7.3f} {r['no_img']:9.3f} {r['no_rna']:7.3f} {r['clinical']:9.3f}", flush=True)
    # derived labels at the quarter ends (the same kind of window as the real label, from the visit scores)
    allmu = predict(model, data, te, device, rows="all", pause=pause)
    q = np.isin(data.prow, te) & (data.kind == QUERY) & data.ym
    out["quarter-end derived labels"] = {"n": int(q.sum()), "MAE": float(np.mean(np.abs(allmu[q] - data.y[q])))}
    print(f"quarter-end derived labels of the held-out patients: n={q.sum()} MAE {out['quarter-end derived labels']['MAE']:.3f}")
    # causality: the state at a step must not change when the later steps are removed
    rng = np.random.default_rng(0)
    pick = [te[j] for j in rng.choice(len(te), min(40, len(te)), replace=False)]
    diffs = []
    for i in pick:
        n = data.n_rows(i)
        k = int(rng.integers(0, n))
        full = allmu[data.off[i] + k]
        sub = Data.__new__(Data)
        sub.__dict__.update(data.__dict__)
        sub.off = np.array([data.off[i], data.off[i] + k + 1])
        sub.kind = data.kind.copy()
        sub.kind[data.off[i] + k] = FINAL  # predict() reads the "final" row of each patient
        diffs.append(abs(predict(model, sub, [0], device)[0][0] - full))
    out["causality_max_diff"] = float(max(diffs))
    print(f"causality check: max |prediction from the whole sequence - from the steps up to it| = {max(diffs):.2e} ({len(pick)} patients)")
    # the quarter-end derived labels of the two test folders (test_2 is the shifted population); they come from the test visits, no
    # target is used
    shift = {}
    for folder in ("test_1", "test_2"):
        t = Data(PATHS[folder], stats=data.stats)
        pats = list(range(t.n_pat))[:max_patients]
        mu_all = predict(model, t, pats, device, rows="all", pause=pause)
        q = np.flatnonzero((t.kind == QUERY) & t.ym & np.isin(t.prow, pats))
        out[f"{folder} quarter-end derived labels"] = {"n": int(len(q)), "MAE": float(np.mean(np.abs(mu_all[q] - t.y[q])))}
        shift.update({f"{folder}_rows": q, f"{folder}_pred": mu_all[q], f"{folder}_y": t.y[q], f"{folder}_patient": t.prow[q]})
        print(f"{folder}: quarter-end derived labels n={len(q)} MAE {out[f'{folder} quarter-end derived labels']['MAE']:.3f}", flush=True)
    np.savez(f"{run_dir}/shift_preds.npz", **shift)
    np.savez(f"{run_dir}/final_preds.npz", patient_id=data.pid[te], y=yt, **{f"{k}_mu": v[0] for k, v in var.items()},
             **{f"{k}_b": v[1] for k, v in var.items()})
    yaml.safe_dump(out, open(f"{run_dir}/metrics.yaml", "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--p_drop", type=float, default=0.5, help="probability to hide a modality of a patient in training")
    ap.add_argument("--emb_pca", type=int, default=0, help="principal components of each embedding as the input (0: the full width, "
                    "the default by user rule; v3 used 16)")
    ap.add_argument("--d_mod", type=int, default=8, help="width of the projection of each modality")
    ap.add_argument("--emb_noise", type=float, default=0.3, help="standard deviation of a noise on the (scaled) embeddings in training")
    ap.add_argument("--v2", type=int, default=1, help="1: head 2 reads the modality features of its step and the observed window share")
    ap.add_argument("--v3", type=int, default=1, help="1: head 2 also learns the real label from the decision step (flag input)")
    ap.add_argument("--partial", type=int, default=1, help="1: head 2 also learns from windows that end after the data")
    ap.add_argument("--rna_weight", type=float, default=3.0, help="weight of the head-2 examples from decision steps with RNAseq")
    ap.add_argument("--plan_per", type=int, default=6, help="candidate (step, horizon) pairs of head 2 per patient and step")
    ap.add_argument("--plan_weight", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.05, help="seconds to wait after each chunk of a prediction")
    ap.add_argument("--train_pause", type=float, default=0.0, help="seconds to wait after each training step (if the GPU runs hot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=int, default=0, help="0: the held-out patients of hs_build.py; k > 0: a new set of the same size "
                    "without them (seed k)")
    ap.add_argument("--fold", default="", help="cross-validation fold f/K (e.g. 2/5): fold 0 is the held-out set of hs_build.py, folds 1..K-1 "
                    "split the other patients; overrides --split")
    ap.add_argument("--drop", default="", help="features removed from the input, comma separated (e.g. ethnicity)")
    ap.add_argument("--raw", type=int, default=1, help="1: no hand-made history summaries in the input (RAW_DROP, RAW_INC); 0: as v3")
    ap.add_argument("--mod_events", type=int, default=0, help="1: an image / RNAseq sample enters the input once, at the first step that "
                    "sees it (0: at every step from its date on)")
    ap.add_argument("--w_recon", type=float, default=0.0, help="weight of the ALDSL view reconstruction: each modality code rebuilds its "
                    "full embedding (0: off)")
    ap.add_argument("--recon_latent", type=int, default=0, help="1: the GRU state (shared by the views) rebuilds both embeddings, also the "
                    "samples hidden by modality dropout (masked modelling)")
    ap.add_argument("--smoke", action="store_true", help="a short test: 1,500 training patients, 150 steps, 300 test patients")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fold = tuple(int(v) for v in args.fold.split("/")) if args.fold else None
    data = Data(PATHS["train"], emb_pca=args.emb_pca, split=args.split, drop=[c for c in args.drop.split(",") if c], fold=fold,
                raw=bool(args.raw), mod_events=bool(args.mod_events))
    data.partial, data.rna_weight, data.real_plan = bool(args.partial), args.rna_weight, bool(args.v3)
    pool = rng.permutation(np.flatnonzero(~data.flag))
    k = int(args.val_frac * len(pool))
    data.val_idx, data.train_idx, data.test_idx = np.sort(pool[:k]), np.sort(pool[k:]), np.flatnonzero(data.flag)
    if args.smoke:
        data.train_idx, data.val_idx = data.train_idx[:1500], data.val_idx[:200]
        args.steps, args.val_every = 150, 75
    run_dir = time.strftime("runs/hs_e2e_%Y%m%d-%H%M%S") + ("_smoke" if args.smoke else "")
    os.makedirs(run_dir)
    yaml.safe_dump({"args": vars(args)}, open(f"{run_dir}/config.yaml", "w"))
    model = make_model(data, vars(args)).to(device)
    print(f"run {run_dir}: {len(data.train_idx)} train, {len(data.val_idx)} validation, {len(data.test_idx)} held-out patients, "
          f"{data.X.shape[1]} features, {sum(p.numel() for p in model.parameters())} parameters, device {device}; split {args.split}, fold {args.fold or 'none'}, "
          f"removed from the input: {data.stats['drop'] or 'none'}", flush=True)
    t0, log = time.time(), open(f"{run_dir}/log.txt", "w")
    step, best_step, best = train(model, data, args, device, run_dir, log)
    torch.save({"model": model.state_dict(), "stats": data.stats, "args": vars(args)}, f"{run_dir}/model.pt")
    print(f"trained {step} steps in {time.time() - t0:.0f}s", flush=True)
    evaluate(model, data, device, run_dir, pause=args.pause, max_patients=300 if args.smoke else None)


if __name__ == "__main__":
    main()
