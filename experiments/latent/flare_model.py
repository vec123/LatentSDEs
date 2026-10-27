"""
Stage 1 flare model (no neural network; CPU): the visit scores as a slow baseline plus flares, fitted by variational inference on the raw
data of the training pool of hs_fused_sde.py (the held-out patients stay out). Ground truth (flare_facts.py): a flare_reactive visit is
an observed flare onset; the excess decays with a half-life of about one month; the level moves over many months.

    log1p(score_j) ~ Laplace(b_i(t_j) + F_i(t_j), sig_y)
    b_i(t)    slow baseline on quarterly knots (linear between them) + the level effect of each biologic; prior (--rw): 2 an integrated
              random walk (second differences, sd sig_b: trends are free, bumps cost, so the bumps belong to the flares), 1 a random walk
    F_i(t)    = sum over the flares k of patient i with tau_k <= t: A_k exp(-(t - tau_k) / D_k)
    onsets    a Poisson process with rate nu_i(t) = exp(a0 + g_t years + c_freq,i + b_nu . drug(t) + biologic(t) gam_nu . rna_i)
              (exact log-likelihood: sum log nu(tau_k) - integral of nu, monthly cells; antibiotics are left out: given in flares)
    log A_k   ~ N(mu_A + c_amp,i + b_A . [drug(tau_k), antibiotic within 7 days of tau_k] + biologic gam_A . rna_i, sig_A)
    log D_k   = d0 + c_dur,i + b_D . [drug(tau_k), antibiotic within 7 days] + biologic gam_D . rna_i   (days)
              (the covariates of amplitude and duration centred over the flares: mu_A and d0 belong to the average flare)
    c_i       = (c_freq, c_amp, c_dur) ~ N(W x_i, diag s^2): x_i the standardized static features, RNAseq PC1-3 and RNAseq presence
              (rna_i = the PCs x presence; RNAseq is sampled at a biologic start)
Variational: Gaussian q for the effect coefficients (prior N(0, 1)), the patient codes, the flare log-amplitudes and the knots; point
estimates for a0, g_t, mu_A, d0 and the scales. The intervals of the effects are mean-field: likely too narrow.
Checks: the decay of the excess after a flare (model against data), and a generation check at the scheduled visits (background,
treatment monitoring): flares drawn from the fitted process against the data, per patient level and oscillation.

    python -m experiments.latent.flare_model [--patients N] [--iters K] [--threads T]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

D = "test_data/hs_public_dataset/public/train/"
NPZ = "test_data/hs_public_dataset/hs_e2e_train.npz"
DRUGS = ["Adalimumab", "Bimekizumab", "Secukinumab"]
YEAR, CELL, KNOT, REACH = 365.25, 30.44, 91.31, 270.0  # days: the rate cells, the baseline knots, the reach of a flare
SCHEDULED = ("background", "treatment_monitoring")


def static_features(dm):
    """standardized static features (one-hot categories against the most frequent one; missing flags for BMI and smoking)"""
    dm = dm.set_index("patient_id")
    cols = {"female": (dm.sex == "Female").astype(float), "age at diagnosis": dm.age_at_diagnosis,
            "hurley stage": dm.hurley_stage_baseline.astype(float), "bmi": dm.bmi_at_baseline.fillna(dm.bmi_at_baseline.median()),
            "bmi missing": dm.bmi_at_baseline.isna().astype(float),
            "diagnostic delay": dm.diagnostic_delay_years.fillna(dm.diagnostic_delay_years.median()),
            "delay missing": dm.diagnostic_delay_years.isna().astype(float), "family history yes": (dm.family_history_hs == "Yes").astype(float)}
    for c, ref in (("ethnicity", "White"), ("smoking_status", "Never")):
        for k in dm[c].fillna("missing").unique():
            if k != ref:
                cols[f"{c.split('_')[0]} {k}"] = (dm[c].fillna("missing") == k).astype(float)
    X = pd.DataFrame(cols)
    return (X - X.mean()) / X.std()


def load(n_patients=None):
    cache = f"runs/flare_model_data{f'_{n_patients}' if n_patients else ''}.pt"  # the arrays of this function (load: ~50 s)
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    d = np.load(NPZ, allow_pickle=True)
    pool = set(d["patient_id"][d["is_test_patient"] == 0])  # the training pool of hs_fused_sde.py (train + validation)
    v = pd.read_csv(D + "visits.csv", parse_dates=["visit_date"])
    v = v[v.patient_id.isin(pool)].sort_values(["patient_id", "visit_date"])
    first = v.groupby("patient_id").visit_date.min()
    v["t"] = (v.visit_date - v.patient_id.map(first)).dt.days.astype(float)
    span = v.groupby("patient_id").t.max()
    pats = np.array(sorted(span.index[span >= 90]))
    if n_patients:
        pats = np.sort(np.random.default_rng(0).choice(pats, n_patients, replace=False))
    v = v[v.patient_id.isin(set(pats))]
    pid = {p: i for i, p in enumerate(pats)}
    tx = pd.read_csv(D + "treatments.csv", parse_dates=["start_date", "end_date"])
    tx = tx[tx.patient_id.isin(set(pats))].copy()
    tx["s"] = (tx.start_date - tx.patient_id.map(first)).dt.days
    tx["e"] = (tx.end_date - tx.patient_id.map(first)).dt.days.fillna(tx.patient_id.map(span))
    bio = {p: [(DRUGS.index(r.drug_name), r.s, r.e) for r in g.itertuples()] for p, g in tx[tx.treatment_class == "Biologic"].groupby("patient_id")}
    ab = {p: np.sort(g.s.to_numpy(float)) for p, g in tx[tx.treatment_class == "Antibiotic"].groupby("patient_id")}

    def drug_at(p, t):
        u = np.zeros((len(t), len(DRUGS)), np.float32)
        for k, s, e in bio.get(p, []):
            u[(t >= s) & (t <= e), k] = 1.0
        return u

    obs, fl, pairs, cells, knots = [], [], [], [], []
    k_off = 0
    for p, g in v.groupby("patient_id", sort=True):
        i = pid[p]
        t, ty, y = g.t.to_numpy(), g.visit_type.to_numpy(), np.log1p(g.ihs4_score.to_numpy(float))
        T = span[p]
        nk = int(T // KNOT) + 2
        tau = t[ty == "flare_reactive"]
        abn = np.array([np.abs(ab[p] - x).min() <= 7 if p in ab else False for x in tau], np.float32)
        f0 = sum(len(x[1]) for x in fl)
        fl.append((np.full(len(tau), i), tau, drug_at(p, tau), abn))
        sc = np.flatnonzero(np.isfinite(y))
        o0 = sum(len(x[1]) for x in obs)
        obs.append((np.full(len(sc), i), t[sc], y[sc], drug_at(p, t[sc]), k_off + (t[sc] // KNOT).astype(int), (t[sc] % KNOT) / KNOT,
                    np.isin(ty[sc], SCHEDULED)))
        lag = t[sc][:, None] - tau[None, :]
        jj, kk = np.nonzero((lag >= 0) & (lag <= REACH))
        pairs.append((o0 + jj, f0 + kk, lag[jj, kk]))
        edges = np.append(np.arange(0, T, CELL), T)
        mid = 0.5 * (edges[1:] + edges[:-1])
        cells.append((np.full(len(mid), i), mid, np.diff(edges), drug_at(p, mid)))
        k0 = np.median(y[sc]) if len(sc) else 1.5
        knots.append((np.full(nk, i), np.full(nk, k0)))
        k_off += nk
    cat = lambda lst, k: np.concatenate([x[k] for x in lst])
    dm = pd.read_csv(D + "demographics.csv")
    X = static_features(dm).loc[pats]
    meta = pd.read_csv(D + "rnaseq_metadata.csv", parse_dates=["collection_date"]).sort_values("collection_date")
    emb = pd.read_csv(D + "rnaseq_embeddings.csv").set_index("rnaseq_sample_id")
    E = emb.to_numpy(np.float64)
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    pcs = pd.DataFrame(U[:, :3] / U[:, :3].std(0), index=emb.index)
    first_rna = meta.drop_duplicates("patient_id").set_index("patient_id").rnaseq_sample_id
    R = np.zeros((len(pats), 3), np.float32)
    has = np.zeros(len(pats), np.float32)
    for p, s in first_rna.items():
        if p in pid:
            R[pid[p]], has[pid[p]] = pcs.loc[s].to_numpy(), 1.0
    Xa = np.column_stack([X.to_numpy(np.float32), R, (has - has.mean()) / has.std()])
    f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    data = dict(
        pats=pats, x_names=list(X.columns) + ["RNAseq PC1", "RNAseq PC2", "RNAseq PC3", "RNAseq sampled"], X=f(Xa), R=f(R * has[:, None]),
        o_p=f(cat(obs, 0), torch.long), o_t=f(cat(obs, 1)), o_y=f(cat(obs, 2)), o_u=f(cat(obs, 3)), o_k=f(cat(obs, 4), torch.long),
        o_f=f(cat(obs, 5)), o_sched=cat(obs, 6),
        f_p=f(cat(fl, 0), torch.long), f_t=f(cat(fl, 1)), f_u=f(cat(fl, 2)), f_ab=f(cat(fl, 3)),
        q_j=f(cat(pairs, 0), torch.long), q_k=f(cat(pairs, 1), torch.long), q_lag=f(cat(pairs, 2)),
        c_p=f(cat(cells, 0), torch.long), c_t=f(cat(cells, 1)), c_dt=f(cat(cells, 2)), c_u=f(cat(cells, 3)),
        k_p=f(cat(knots, 0), torch.long), k_init=f(cat(knots, 1)))
    kp = data["k_p"]
    data["k_next"] = torch.nonzero(kp[1:] == kp[:-1]).ravel()  # knot m and m + 1 of the same patient
    data["k_mid"] = torch.nonzero((kp[1:-1] == kp[:-2]) & (kp[1:-1] == kp[2:])).ravel() + 1  # knots m - 1, m, m + 1 of the same patient
    os.makedirs("runs", exist_ok=True)
    torch.save(data, cache)
    return data


class FlareModel(torch.nn.Module):
    BLOCKS = {"b_nu": 3, "b_A": 4, "b_D": 4, "b_L": 3, "g_R": 9}

    def __init__(self, data, rw=2, nc=False):
        super().__init__()
        n, K, Q, px = len(data["pats"]), len(data["f_t"]), len(data["k_init"]), data["X"].shape[1]
        self.rw = rw
        # nc (non-centred): c = W x + s eta and log A = prior mean + sig_A eps, with q over eta and eps (prior N(0, 1)); the centred form
        # (q over c and log A) shrank the variances about 5x in the recovery test (flare_recovery.py)
        self.nc = nc
        # the covariates of amplitude and duration centred over the flares: mu_A and d0 belong to the average flare
        self.register_buffer("v_mean", torch.cat([data["f_u"], data["f_ab"][:, None]], 1).mean(0))
        P = torch.nn.Parameter
        self.a0, self.g_t = P(torch.tensor(np.log(1.85), dtype=torch.float32)), P(torch.tensor(0.0))
        self.mu_A, self.ls_A, self.d0 = P(torch.tensor(np.log(0.25), dtype=torch.float32)), P(torch.tensor(-0.2)), P(torch.tensor(np.log(40.0), dtype=torch.float32))
        self.ls_b, self.ls_y, self.ls_c = P(torch.tensor(np.log(0.15), dtype=torch.float32)), P(torch.tensor(np.log(0.2), dtype=torch.float32)), P(torch.full((3,), np.log(0.3)))
        self.n_th = sum(self.BLOCKS.values()) + 3 * px
        self.th_m, self.th_l = P(torch.zeros(self.n_th)), P(torch.full((self.n_th,), -3.0))
        self.c_m, self.c_l = P(torch.zeros(n, 3)), P(torch.full((n, 3), -2.0))
        self.a_m, self.a_l = P(torch.full((K,), 0.0 if nc else float(np.log(0.25)))), P(torch.full((K,), -1.0))
        self.k_m, self.k_l = P(data["k_init"].clone()), P(torch.full((Q,), -2.0))
        self.px = px

    def split(self, th):
        out, a = {}, 0
        for k, n in self.BLOCKS.items():
            out[k], a = th[a:a + n], a + n
        out["g_R"] = out["g_R"].view(3, 3)  # rows: PC1-3; columns: rate, amplitude, duration
        out["W"] = th[a:].view(3, self.px)
        return out

    @staticmethod
    def kl_normal(m, ls, pm, ps_log):
        return (ps_log - ls + ((2 * ls).exp() + (m - pm) ** 2) / (2 * (2 * ps_log).exp()) - 0.5).sum()

    def parts(self, d, sample=True):
        e = (lambda x: torch.randn_like(x)) if sample else torch.zeros_like
        th = self.split(self.th_m + self.th_l.exp() * e(self.th_m))
        c = self.c_m + self.c_l.exp() * e(self.c_m)
        if self.nc:
            c = d["X"] @ th["W"].T + self.ls_c.exp() * c
        kn = self.k_m + self.k_l.exp() * e(self.k_m)
        rna = d["R"] @ th["g_R"]  # (n, 3): the biologic-response shift of rate, amplitude, duration by RNAseq
        bio_f = (d["f_u"].sum(1) > 0).float()
        V = torch.cat([d["f_u"], d["f_ab"][:, None]], 1) - self.v_mean
        fp = d["f_p"]
        lognu_on = self.a0 + self.g_t * d["f_t"] / YEAR + c[fp, 0] + d["f_u"] @ th["b_nu"] + bio_f * rna[fp, 0]
        bio_c = (d["c_u"].sum(1) > 0).float()
        cp = d["c_p"]
        lognu_c = self.a0 + self.g_t * d["c_t"] / YEAR + c[cp, 0] + d["c_u"] @ th["b_nu"] + bio_c * rna[cp, 0]
        prior_A = self.mu_A + c[fp, 1] + V @ th["b_A"] + bio_f * rna[fp, 1]
        logA = self.a_m + self.a_l.exp() * e(self.a_m)
        if self.nc:
            logA = prior_A + self.ls_A.exp() * logA
        logD = self.d0 + c[fp, 2] + V @ th["b_D"] + bio_f * rna[fp, 2]
        F = torch.zeros_like(d["o_y"]).index_add_(0, d["q_j"], (logA[d["q_k"]] - d["q_lag"] / logD[d["q_k"]].exp()).exp())
        b = (1 - d["o_f"]) * kn[d["o_k"]] + d["o_f"] * kn[d["o_k"] + 1] + d["o_u"] @ th["b_L"]
        return dict(th=th, c=c, logA=logA, lognu_on=lognu_on, lognu_c=lognu_c, prior_A=prior_A, logD=logD, F=F, b=b)

    def loss(self, d):
        r = self.parts(d)
        th = r["th"]
        ll_rate = r["lognu_on"].sum() - (d["c_dt"] / YEAR * r["lognu_c"].exp()).sum()
        ll_obs = (-np.log(2.0) - self.ls_y - (d["o_y"] - r["b"] - r["F"]).abs() / self.ls_y.exp()).sum()
        v = (2 * self.k_l).exp()
        if self.rw == 1:  # first differences: a random walk (level steps)
            k1, k2 = d["k_next"], d["k_next"] + 1
            e2 = (self.k_m[k2] - self.k_m[k1]) ** 2 + v[k1] + v[k2]
        else:  # second differences: an integrated random walk (trends are free, bumps cost: the bumps belong to the flares)
            k0, k1, k2 = d["k_mid"] - 1, d["k_mid"], d["k_mid"] + 1
            e2 = (self.k_m[k2] - 2 * self.k_m[k1] + self.k_m[k0]) ** 2 + v[k0] + 4 * v[k1] + v[k2]
        ll_rw = (-self.ls_b - 0.5 * np.log(2 * np.pi) - e2 / (2 * (2 * self.ls_b).exp())).sum() + self.k_l.sum()
        if self.nc:
            kl_A = self.kl_normal(self.a_m, self.a_l, torch.zeros_like(self.a_m), torch.zeros_like(self.a_m))
            kl_c = self.kl_normal(self.c_m, self.c_l, torch.zeros_like(self.c_m), torch.zeros_like(self.c_m))
        else:
            kl_A = self.kl_normal(self.a_m, self.a_l, r["prior_A"], self.ls_A)
            kl_c = self.kl_normal(self.c_m, self.c_l, d["X"] @ th["W"].T, self.ls_c.expand_as(self.c_m))
        kl_th = self.kl_normal(self.th_m, self.th_l, torch.zeros_like(self.th_m), torch.zeros_like(self.th_m))
        n = len(d["o_y"])
        tot = -(ll_rate + ll_obs + ll_rw - kl_A - kl_c - kl_th) / n
        return tot, {k: float(x) / n for k, x in (("rate", ll_rate), ("obs", ll_obs), ("rw", ll_rw), ("kl_A", kl_A), ("kl_c", kl_c))}


def fit(model, d, iters, log):
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0 if k < 0.6 * iters else 0.2)
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        loss, parts = model.loss(d)
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == iters - 1:
            log(f"{it:5d} t={time.time() - t0:5.0f}s loss {float(loss):.4f} " + " ".join(f"{k} {v:+.4f}" for k, v in parts.items()) +
                f" | rate/yr {model.a0.exp():.2f} half-life {np.log(2) * model.d0.exp():.0f}d sig_A {model.ls_A.exp():.2f} "
                f"sig_b {model.ls_b.exp():.3f} sig_y {model.ls_y.exp():.3f} s_c " + " ".join(f"{x:.2f}" for x in model.ls_c.exp().tolist()))


def report(model, d, log):
    th = model.split(model.th_m.detach())
    sd = model.split(model.th_l.detach().exp())
    ci = lambda m, s: f"{np.exp(m):5.2f} [{np.exp(m - 1.96 * s):5.2f}, {np.exp(m + 1.96 * s):5.2f}]"
    log(f"\nGlobal: flare rate {model.a0.exp():.2f} per year at the first visit (no biologic, average patient), x{model.g_t.exp():.3f} per "
        f"year of follow-up; average flare: amplitude factor x{np.exp(np.exp(float(model.mu_A))):.2f} on (1 + score) at the onset (sd of "
        f"log A {model.ls_A.exp():.2f}), half-life {np.log(2) * model.d0.exp():.0f} days; baseline ({'level steps' if model.rw == 1 else 'trend changes'}) "
        f"sd {model.ls_b.exp():.3f} per quarter (log1p); visit noise (Laplace scale) {model.ls_y.exp():.3f}")
    log("\nDrug effects (posterior mean factor [95% mean-field interval]); level: factor on (1 + baseline score)")
    log(f"   {'drug':14s} {'flare rate':>22s} {'amplitude (log A)':>22s} {'duration':>22s} {'level':>22s}")
    for k, g in enumerate(DRUGS):
        log(f"   {g:14s} {ci(th['b_nu'][k], sd['b_nu'][k]):>22s} {ci(th['b_A'][k], sd['b_A'][k]):>22s} {ci(th['b_D'][k], sd['b_D'][k]):>22s} "
            f"{ci(th['b_L'][k], sd['b_L'][k]):>22s}")
    log(f"   {'antibiotic':14s} {'(left out: given in flares)':>22s} {ci(th['b_A'][3], sd['b_A'][3]):>22s} {ci(th['b_D'][3], sd['b_D'][3]):>22s}"
        f"   (antibiotic started within 7 days of the onset: reverse-causal for the amplitude)")
    log("\nRNAseq x biologic (factor per SD of the PC while on a biologic; RNAseq patients only)")
    for k in range(3):
        log(f"   PC{k + 1}: rate {ci(th['g_R'][k, 0], sd['g_R'][k, 0])}  amplitude {ci(th['g_R'][k, 1], sd['g_R'][k, 1])}  duration "
            f"{ci(th['g_R'][k, 2], sd['g_R'][k, 2])}")
    s = model.ls_c.exp().detach()
    mx = d["X"] @ th["W"].T
    vx = mx.var(0)
    log("\nPatient traits: sd of the code beyond the features (factor between the 90th and 10th percentile patient), share of the "
        "variance the features explain")
    for k, name in enumerate(("frequency", "amplitude", "duration")):
        log(f"   {name:10s} sd {s[k]:.2f} (x{np.exp(2 * 1.2816 * float(s[k])):.2f}); features explain {float(vx[k] / (vx[k] + s[k] ** 2)):.0%}; "
            f"total sd between patients {float((vx[k] + s[k] ** 2).sqrt()):.2f}"
            + ("   (sd < 0.05: the variation beyond the features collapsed, not identified)" if s[k] < 0.05 else ""))
    with torch.no_grad():
        cm = model.parts(d, sample=False)["c"].numpy()  # the posterior mean of the codes (either form)
    log("   correlation of the posterior codes (frequency-amplitude, frequency-duration, amplitude-duration): " +
        ", ".join(f"{np.corrcoef(cm[:, a], cm[:, b])[0, 1]:+.2f}" for a, b in ((0, 1), (0, 2), (1, 2))))
    log("\nFeature effects on the traits (factor per SD of the feature [95% mean-field interval])")
    log(f"   {'feature':28s} {'frequency':>22s} {'amplitude':>22s} {'duration':>22s}")
    for j, nm in enumerate(d["x_names"]):
        log(f"   {nm:28s} " + " ".join(f"{ci(th['W'][k, j], sd['W'][k, j]):>22s}" for k in range(3)))


@torch.no_grad()
def checks(model, d, log, seed=0):
    r = model.parts(d, sample=False)
    # 1. decay of the excess: data (flare score - baseline >= 3 on the score scale) against the model at the same visit pairs
    y, b, F = d["o_y"].numpy(), r["b"].numpy(), r["F"].numpy()
    j, k, lag = d["q_j"].numpy(), d["q_k"].numpy(), d["q_lag"].numpy()
    fp, ft, ot = d["f_p"].numpy(), d["f_t"].numpy(), d["o_t"].numpy()
    nxt = np.where(np.append(fp[1:] == fp[:-1], False), np.append(ft[1:], np.inf), np.inf)  # the next onset of the same patient
    onset = {int(kk): int(jj) for jj, kk, ll in zip(j, k, lag) if ll == 0}
    rows = []
    for jj, kk, ll in zip(j, k, lag):
        if ll > 0 and int(kk) in onset and ot[jj] < nxt[kk]:
            j0 = onset[int(kk)]
            base = np.expm1(b[j0])
            amp = np.expm1(y[j0]) - base
            if amp >= 3:
                rows.append((ll, (np.expm1(y[jj]) - base) / amp, (np.expm1(b[jj] + F[jj]) - base) / amp,
                             np.exp(-ll / np.exp(float(r["logD"][kk])))))
    rr = pd.DataFrame(rows, columns=["lag", "data", "fit", "flare"])
    rr["bin"] = pd.cut(rr.lag, [0, 10, 20, 35, 60, 90, 180, 270])
    log("\nCheck 1: excess after a flare relative to the onset excess (visits before the next flare; baseline = fitted b at the onset):"
        " data, the full fit (baseline + all flares), the decay of this flare alone (medians)")
    for bb, g in rr.groupby("bin", observed=True):
        log(f"   days {str(bb):10s} n={len(g):6d}  data {g.data.median():+.2f}  fit {g.fit.median():+.2f}  flare alone {g.flare.median():+.2f}")
    # 2. generation at the scheduled visits: flares drawn from the fitted process (posterior mean codes, own drug path), noise drawn
    rng = np.random.default_rng(seed)
    c = r["c"].numpy()
    th = r["th"]
    lam = (r["lognu_c"].exp() * d["c_dt"] / YEAR).numpy()
    cp, ct, cdt, cu = d["c_p"].numpy(), d["c_t"].numpy(), d["c_dt"].numpy(), d["c_u"].numpy()
    n_on = rng.poisson(lam)
    sp = np.repeat(cp, n_on)
    st = np.repeat(ct - cdt / 2, n_on) + rng.random(n_on.sum()) * np.repeat(cdt, n_on)
    su = np.repeat(cu, n_on, 0)
    sab = (rng.random(len(st)) < float(d["f_ab"].mean())).astype(np.float32)
    rna = (d["R"] @ th["g_R"]).numpy()
    bio = (su.sum(1) > 0).astype(np.float32)
    V = np.column_stack([su, sab]) - model.v_mean.numpy()
    sA =float(model.mu_A) + c[sp, 1] + V @ th["b_A"].numpy() + bio * rna[sp, 1] + float(model.ls_A.exp()) * rng.standard_normal(len(st))
    sD = np.exp(float(model.d0) + c[sp, 2] + V @ th["b_D"].numpy() + bio * rna[sp, 2])
    sched = d["o_sched"]
    op, ot = d["o_p"].numpy(), d["o_t"].numpy()
    order = np.argsort(sp, kind="stable")
    sp, st, sA, sD = sp[order], st[order], sA[order], sD[order]
    starts = np.searchsorted(sp, np.arange(len(d["pats"]) + 1))
    Fs = np.zeros(len(y))
    for jj in np.flatnonzero(sched):
        i = op[jj]
        a, z = starts[i], starts[i + 1]
        lg = ot[jj] - st[a:z]
        m = (lg >= 0) & (lg <= REACH)
        Fs[jj] = np.sum(np.exp(sA[a:z][m] - lg[m] / sD[a:z][m]))
    noise = rng.laplace(0, float(model.ls_y.exp()), len(y))
    sim = np.expm1(b + Fs + noise).clip(min=0)
    sim0 = np.expm1(b + noise).clip(min=0)
    obs = np.expm1(y)
    out = []
    for i in np.unique(op[sched]):
        jj = np.flatnonzero(sched & (op == i))
        if len(jj) >= 4:
            out.append([obs[jj].mean(), np.abs(np.diff(obs[jj])).mean(), sim[jj].mean(), np.abs(np.diff(sim[jj])).mean(),
                        sim0[jj].mean(), np.abs(np.diff(sim0[jj])).mean()])
    o = np.array(out)
    sp_ = lambda a, b_: pd.Series(a).corr(pd.Series(b_), method="spearman")
    log(f"\nCheck 2: generation at the scheduled visits ({len(o)} patients with >= 4 scored scheduled visits; flares drawn, not the observed ones)")
    log(f"   level (mean score)            data median {np.median(o[:, 0]):5.2f}   with flares {np.median(o[:, 2]):5.2f} (Spearman with data "
        f"{sp_(o[:, 0], o[:, 2]):.2f})   without flares {np.median(o[:, 4]):5.2f}")
    log(f"   oscillation (mean |change|)   data median {np.median(o[:, 1]):5.2f}   with flares {np.median(o[:, 3]):5.2f} (Spearman with data "
        f"{sp_(o[:, 1], o[:, 3]):.2f})   without flares {np.median(o[:, 5]):5.2f}")
    log(f"   drawn flares per patient-year {len(st) / (d['c_dt'].sum().item() / YEAR):.2f} (data {len(d['f_t']) / (d['c_dt'].sum().item() / YEAR):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=0, help="a random subset (0: the whole training pool)")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rw", type=int, default=2, help="baseline prior: 1 random walk (level steps), 2 integrated random walk (trend changes)")
    ap.add_argument("--nc", type=int, default=0, help="1: non-centred codes and flare amplitudes (see FlareModel)")
    ap.add_argument("--out", default="runs/flare_model")
    ap.add_argument("--eval", type=int, default=0, help="1: no fit, the report and the checks of <out>/model.pt again")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    d = load(a.patients or None)
    log(f"{len(d['pats'])} patients, {len(d['o_y'])} scored visits, {len(d['f_t'])} flare onsets, {len(d['q_j'])} visit-flare pairs, "
        f"{len(d['c_t'])} rate cells, {len(d['k_init'])} knots; loaded in {time.time() - t0:.0f}s")
    if a.eval:
        ck = torch.load(f"{a.out}/model.pt", weights_only=False)
        model = FlareModel(d, rw=ck["rw"], nc=ck.get("nc", False))
        model.load_state_dict(ck["model"])
    else:
        model = FlareModel(d, rw=a.rw, nc=bool(a.nc))
        fit(model, d, a.iters, log)
    report(model, d, log)
    checks(model, d, log)
    torch.save({"model": model.state_dict(), "pats": d["pats"], "x_names": d["x_names"], "rw": model.rw, "nc": model.nc},
               f"{a.out}/model.pt")
    with open(f"{a.out}/report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
