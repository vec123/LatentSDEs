"""
ALDSL (Chen et al., "Adaptive latent disease state learning ...", Pattern Recognition 172 (2026) 112389; code in challenge_references/ALDSL)
on the HS data. ALDSL is unsupervised multi-view feature selection: each view X^v is denoised (U^v, with a learned graph of its variables),
projected to a shared latent disease state H (H^T H = I) by P^v, and a variable is ranked by the row norm of P^v.

Views: clinical (the final-step features at the cutoff of hs_e2e_train.npz, without ethnicity), image embedding (256), RNAseq embedding (128).
Cohort: the 2,800 train-folder patients with an image; 600 of them also have RNAseq, so the RNAseq view is missing for 79%.

Port of the reference code (main.py, the settings of its __main__ are the defaults here) with three changes:
  - Euclidean loss for all views (the L1 / ADMM branch of the code is for count-valued SNP data; all views here are continuous);
  - missing views as in eq. (9)-(10) of the paper (the code has no M^v): view v holds only its n^v patients, its projection term reads H at
    their rows and has the weight alpha * n / n^v (= alpha for a complete view); the H update stays an orthogonal Procrustes step;
  - thin SVD in the H update (the code takes the full n x n SVD);
  - 60 iterations, not 200: the objective is flat after ~50, and the top 10 of each view after 30, 60 and 100 iterations equal those
    after 200 (whole-cohort fit, 23 s for 200 iterations).

A. avg_ihs4_next6 (the 2026 label; exercises 1 and 3), paper protocol: top k variables per view -> downstream SVR (the paper's model) and
   HistGradientBoosting (L1). Folds of the GRU cross-validation (runs/cv5_runs.txt); ALDSL and every selection are refit inside each training
   fold. Reference sets: all dimensions, 16 PCs (the GRU input), random k, top k by |corr| with the label (supervised), the latent H.
   Paired against the GRU and XGBoost CV predictions of the same patients.
B. Good responder at a biologic start with a baseline RNAseq sample (131 starts, the definitions of rna_profiles.py): logistic regression on
   the selected RNAseq dimensions, out-of-fold AUC (5 x 5-fold). The cutoff views include visits after the start, so the selection is also
   made from a fit on the decision step (the day before the start) of the 600 RNAseq patients (no information after the start).
C. Selection: stability over the folds, the chosen clinical variables, ALDSL against ALDSL_C (complete cases only), diagnostics.

    python -I experiments/latent/aldsl_hs.py [--k 10] [--folds 0,1,2,3,4] [--models svr,hgb] [--out runs/aldsl_hs.txt]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

TRAIN = "test_data/hs_public_dataset/hs_e2e_train.npz"
RAW = "test_data/hs_public_dataset/public/train/"
FINAL, DECISION = 2, 3
DROP = {"is_query", "is_final", "is_decision", "ethnicity", "img_days_since", "rna_days_since"}
VIEWS = ("clinical", "image", "RNAseq")
HGB_SETS = ("clinical (all)", "clinical + 16 PCs each", "clinical + ALDSL top", "clinical + ALDSL latent", "clinical + random", "clinical + |corr|",
            "ALDSL top", "ALDSL latent", "PCs of all views")  # HGB only on these (time)
T0 = time.time()
warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanstd / nanmean of the all-missing clinical columns
lines = []


def out(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def log(msg):
    out(f"[{time.time() - T0:5.0f}s] {msg}")


def boot(e, n=2000, seed=0):
    r = np.random.default_rng(seed)
    return np.percentile(e[r.integers(0, len(e), (n, len(e)))].mean(1), [2.5, 97.5])


def sq_dist_cols(U):
    """squared Euclidean distances between the columns (variables) of U"""
    g = (U * U).sum(0)
    return np.maximum(g[:, None] + g[None, :] - 2 * U.T @ U, 0)


class ALDSL:
    def __init__(self, alpha=100.0, beta=1e6, gamma=10.0, h=5, xi=1.0, pi=5.0, n_iter=60, seed=0):
        self.alpha, self.beta, self.gamma, self.h, self.xi, self.pi, self.n_iter, self.seed = alpha, beta, gamma, h, xi, pi, n_iter, seed

    def fit(self, views, rows, n):
        """views[v]: n^v x d^v (standardized, no NaN); rows[v]: the rows of these patients in 0..n-1"""
        rng = np.random.default_rng(self.seed)
        V, h, xi, pi = len(views), self.h, self.xi, self.pi
        a = np.array([self.alpha * n / len(r) for r in rows])
        lam = np.ones(V) / V
        U = [X.copy() for X in views]
        P = [rng.standard_normal((X.shape[1], h)) for X in views]
        b = [np.zeros(h) for _ in views]
        L = []
        for X in views:
            S = np.full((X.shape[1],) * 2, 1.0 / X.shape[1]) ** 2
            L.append(np.diag(S.sum(1)) - S)
        H = rng.standard_normal((n, h))
        self.obj = []
        for it in range(self.n_iter):
            Hv = [H[r] - b[v] for v, r in enumerate(rows)]
            for v, X in enumerate(views):  # U: the denoised view
                A = np.eye(X.shape[1]) + 2 * self.gamma * L[v] + a[v] * P[v] @ P[v].T
                U[v] = cho_solve(cho_factor(A), (X + a[v] * Hv[v] @ P[v].T).T).T
            for v in range(V):  # S: the graph of the variables, s_ij ~ 1 / ||U_:i - U_:j||^2, rows sum to 1
                Sd = 1.0 / (sq_dist_cols(U[v]) + 1e-12)
                np.fill_diagonal(Sd, 0.0)
                S = (Sd / Sd.sum(1, keepdims=True)) ** 2
                S = (S + S.T) / 2
                L[v] = np.diag(S.sum(1)) - S
            for v in range(V):  # P: projection with the adaptive regularization ||P||_xi (reweighted ridge)
                w = np.linalg.norm(P[v], axis=1)
                Dg = 0.5 * (1 + xi) * (w + 2 * xi) / ((w + xi) ** 2 + 1e-12)
                P[v] = cho_solve(cho_factor(U[v].T @ U[v] + self.beta / a[v] * np.diag(Dg)), U[v].T @ Hv[v])
            for v, r in enumerate(rows):
                b[v] = (H[r] - U[v] @ P[v]).mean(0)
            M = np.zeros((n, h))  # H: orthogonal Procrustes on the weighted sum of the views' projections
            for v, r in enumerate(rows):
                M[r] += lam[v] ** pi * a[v] * (U[v] @ P[v] + b[v])
            W, s, Vt = np.linalg.svd(M, full_matrices=False)
            H = W @ Vt
            loss = []
            for v, (X, r) in enumerate(zip(views, rows)):
                w = np.linalg.norm(P[v], axis=1)
                loss.append(np.sum((X - U[v]) ** 2) + self.gamma * np.sum((U[v] @ L[v]) * U[v])
                            + a[v] * np.sum((H[r] - b[v] - U[v] @ P[v]) ** 2) + self.beta * np.sum((1 + xi) * w * w / (w + xi)))
            loss = np.array(loss)
            self.obj.append(float(np.sum(lam ** pi * loss)))
            lam = (pi * loss) ** (1 / (1 - pi))
            lam = lam / lam.sum()
        self.U, self.P, self.b, self.H, self.lam, self.a, self.Dg_last = U, P, b, H, lam, a, Dg
        self.T = Vt.T @ np.diag(1 / s) @ Vt  # H = M T: the same map for new patients
        self.dg_spread = []
        for v in range(V):
            w = np.linalg.norm(P[v], axis=1)
            Dg = 0.5 * (1 + xi) * (w + 2 * xi) / ((w + xi) ** 2 + 1e-12)
            self.dg_spread.append((float(w.max()), float(Dg.max() / Dg.min())))
        return self

    def rank(self, v):
        return np.argsort(-np.linalg.norm(self.P[v], axis=1))

    def latent(self, views, rows, n, only=None):
        """H of any patients from their raw views (the H update with X in place of U); only = a list of views to use"""
        M = np.zeros((n, self.h))
        for v, r in enumerate(rows):
            if only is None or v in only:
                M[r] += self.lam[v] ** self.pi * self.a[v] * (views[v] @ self.P[v] + self.b[v])
        return M @ self.T


def standardize(A, fit_rows):
    mu, sd = np.nanmean(A[fit_rows], 0), np.nanstd(A[fit_rows], 0)
    sd[~(sd > 0)] = 1.0
    return np.nan_to_num((A - mu) / sd)


def abs_corr(A, y):
    A = A - A.mean(0)
    yc = y - y.mean()
    return np.abs(A.T @ yc) / (np.sqrt((A * A).sum(0) * (yc @ yc)) + 1e-12)


def jaccard_mean(sets):
    J = [len(set(a) & set(b)) / len(set(a) | set(b)) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(J))


def fit_predict(model, Atr, ytr, Ate):
    if model == "svr":
        return SVR(C=10.0, epsilon=0.5, gamma="scale").fit(Atr, ytr).predict(Ate)
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.06, max_iter=400, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=25, random_state=0).fit(Atr, ytr).predict(Ate)


def load():
    d = np.load(TRAIN)
    names, kind, off = [str(x) for x in d["feature_names"]], d["kind"], d["offsets"]
    X, Mk = d["X"], d["M"]
    pid = d["patient_id"]
    fr = np.flatnonzero(kind == FINAL)
    assert len(fr) == len(pid) and np.all((fr >= off[:-1]) & (fr < off[1:]))
    clin = [j for j, c in enumerate(names) if c not in DROP]
    C = np.where(Mk[fr][:, clin] > 0, X[fr][:, clin], np.nan).astype(float)
    img_row, rna_row = d["img_row"], d["rna_row"]
    img_emb, rna_emb = d["img_emb"].astype(float), d["rna_emb"].astype(float)
    y, lab = d["y"][fr].astype(float), d["y_mask"][fr] > 0
    # decision steps that see a baseline RNAseq sample (taken on the start day)
    j_rds = names.index("rna_days_since")
    dec = np.flatnonzero((kind == DECISION) & (rna_row >= 0))
    dec = dec[(Mk[dec, j_rds] > 0) & (X[dec, j_rds] == 0)]
    dec_pat = np.searchsorted(off, dec, side="right") - 1
    _, first = np.unique(dec_pat, return_index=True)
    dec, dec_pat = dec[first], dec_pat[first]
    D = dict(pid=pid, names=[names[j] for j in clin], C=C, y=y, lab=lab,
             I=np.where((img_row[fr] >= 0)[:, None], img_emb[np.maximum(img_row[fr], 0)], np.nan),
             R=np.where((rna_row[fr] >= 0)[:, None], rna_emb[np.maximum(rna_row[fr], 0)], np.nan), rna_emb=rna_emb,
             dec_pat=dec_pat, dec_C=np.where(Mk[dec][:, clin] > 0, X[dec][:, clin], np.nan).astype(float),
             dec_I=np.where((img_row[dec] >= 0)[:, None], img_emb[np.maximum(img_row[dec], 0)], np.nan),
             dec_R=rna_emb[rna_row[dec]])
    pos = {p: i for i, p in enumerate(pid)}
    fold, gru, xgb = (np.full(len(pid), np.nan) for _ in range(3))
    for ln in open("runs/cv5_runs.txt"):
        p = ln.split()
        f, r = int(p[1]), p[5]
        g = np.load(f"{r}/final_preds.npz")
        idx = np.array([pos[q] for q in g["patient_id"]])
        assert np.allclose(g["y"], y[idx])
        fold[idx], gru[idx] = f, g["full_mu"]
        o = np.load(f"runs/xgb_e2e_cv5_fold{f}/final_preds.npz")
        xgb[np.array([pos[q] for q in o["patient_id"]])] = o["full_mu"]
    D.update(fold=fold, gru=gru, xgb=xgb)
    return D


def responder_starts():
    """the starts of rna_profiles.py: a baseline RNAseq sample on the start day; ratio = mean score days 90..270 after / mean of the 180 days
    before (pre mean >= 3); good responder: ratio < 0.5"""
    t = pd.read_csv(RAW + "treatments.csv", parse_dates=["start_date"])
    v = pd.read_csv(RAW + "visits.csv", parse_dates=["visit_date"]).dropna(subset=["ihs4_score"])
    demo = pd.read_csv(RAW + "demographics.csv").set_index("patient_id")
    rm = pd.read_csv(RAW + "rnaseq_metadata.csv", parse_dates=["collection_date"])
    b = t[t.treatment_class == "Biologic"].merge(rm[["patient_id", "collection_date"]], on="patient_id")
    b = b[b.start_date == b.collection_date].drop_duplicates("patient_id")
    sc = {p: g for p, g in v.groupby("patient_id")}
    rows = []
    for r in b.itertuples():
        g = sc.get(r.patient_id)
        if g is None:
            continue
        dd = (g.visit_date - r.start_date).dt.days
        pre, post = g.ihs4_score[(dd >= -180) & (dd < 0)], g.ihs4_score[(dd >= 90) & (dd <= 270)]
        if len(pre) == 0 or len(post) == 0 or pre.mean() < 3:
            continue
        prior = g.ihs4_score[dd < 0]
        dm = demo.loc[r.patient_id]
        rows.append(dict(patient_id=r.patient_id, ratio=post.mean() / pre.mean(), B=pre.mean(), n_pre=len(pre), last_pre=prior.iloc[-1],
                         mean_all_pre=prior.mean(), age=r.start_date.year - dm.birth_year, sex=float(dm.sex == "Female"), bmi=dm.bmi_at_baseline,
                         hurley=dm.hurley_stage_baseline, delay=dm.diagnostic_delay_years, line=r.treatment_line, drug=r.drug_name))
    return pd.DataFrame(rows)


def run_fold(f, Cr, I, R, y, fold, has_r, k, models, n_iter, h, only):
    """ALDSL, ALDSL_C and every selection fitted on the training patients of fold f; the downstream predictions of its held-out patients"""
    nc = len(y)
    trm = fold != f
    tem = ~trm
    Cs, Is = standardize(Cr, trm), standardize(I, trm)
    Rs = standardize(R, trm & has_r)
    Rs[~has_r] = 0.0
    flag = has_r[:, None].astype(float)
    rtr = np.flatnonzero(has_r[trm])
    m = ALDSL(n_iter=n_iter, h=h).fit([Cs[trm], Is[trm], Rs[trm][rtr]], [np.arange(trm.sum()), np.arange(trm.sum()), rtr], int(trm.sum()))
    cc = trm & has_r
    mc = ALDSL(n_iter=n_iter, h=h).fit([Cs[cc], Is[cc], Rs[cc]], [np.arange(cc.sum())] * 3, int(cc.sum()))
    rk = [m.rank(v) for v in range(3)]
    rkc = [mc.rank(v) for v in range(3)]
    rco = [np.argsort(-abs_corr(Cs[trm], y[trm])), np.argsort(-abs_corr(Is[trm], y[trm])), np.argsort(-abs_corr(Rs[cc], y[cc]))]
    Hc = standardize(m.latent([Cs, Is, Rs[has_r]], [np.arange(nc), np.arange(nc), np.flatnonzero(has_r)], nc), trm)  # H ~ 1/sqrt(n)
    pi_ = PCA(16, random_state=0).fit(Is[trm])
    pr_ = PCA(16, random_state=0).fit(Rs[cc])
    Ip, Rp = pi_.transform(Is), np.where(has_r[:, None], pr_.transform(Rs), 0.0)
    All = np.column_stack([Cs, Is, Rs, flag])
    Ap = standardize(PCA(h, random_state=0).fit(All[trm]).transform(All), trm)  # unfused reference of the same size as H
    sets = {
        "clinical (all)": Cs,
        "clinical + image + RNAseq (all dims)": np.column_stack([Cs, Is, Rs, flag]),
        "clinical + 16 PCs each (GRU input)": np.column_stack([Cs, Ip, Rp, flag]),
        f"clinical + ALDSL top {k} image / RNAseq": np.column_stack([Cs, Is[:, rk[1][:k]], Rs[:, rk[2][:k]], flag]),
        f"clinical + ALDSL_C top {k} image / RNAseq": np.column_stack([Cs, Is[:, rkc[1][:k]], Rs[:, rkc[2][:k]], flag]),
        f"clinical + |corr| top {k} (supervised)": np.column_stack([Cs, Is[:, rco[1][:k]], Rs[:, rco[2][:k]], flag]),
        f"clinical + ALDSL latent H ({m.h})": np.column_stack([Cs, Hc]),
        f"ALDSL latent H ({m.h}) alone": Hc,
        f"PCs of all views concatenated ({h}, alone)": Ap,
        f"ALDSL top {k} per view (paper protocol)": np.column_stack([Cs[:, rk[0][:k]], Is[:, rk[1][:k]], Rs[:, rk[2][:k]], flag]),
        f"|corr| top {k} per view (supervised)": np.column_stack([Cs[:, rco[0][:k]], Is[:, rco[1][:k]], Rs[:, rco[2][:k]], flag]),
    }
    rng = np.random.default_rng(100 + f)
    for dr in range(3):
        ji, jr, jc = rng.choice(256, k, replace=False), rng.choice(128, k, replace=False), rng.choice(Cr.shape[1], k, replace=False)
        sets[f"clinical + random {k} image / RNAseq #{dr}"] = np.column_stack([Cs, Is[:, ji], Rs[:, jr], flag])
        sets[f"random {k} per view #{dr}"] = np.column_stack([Cs[:, jc], Is[:, ji], Rs[:, jr], flag])
    pf = {}
    for name, A in sets.items():
        for mo in models:
            if (mo == "hgb" and not name.startswith(HGB_SETS)) or (only and not name.startswith(only)):
                continue
            pf[(name, mo)] = fit_predict(mo, A[trm], y[trm], A[tem])
    sf = {"ALDSL": [r[:k] for r in rk], "ALDSL_C": [r[:k] for r in rkc], "corr": [r[:k] for r in rco]}
    msg = (f"ALDSL lambda " + " / ".join(f"{x:.3f}" for x in m.lam) + f"; objective {m.obj[0]:.4g} -> {m.obj[-1]:.4g}; "
           "max row norm of P / xi, max/min reweight: " + ", ".join(f"{w:.1e} {sp:.4f}" for w, sp in m.dg_spread))
    return tem, pf, sf, m.lam, msg


def task_a(D, args):
    k = args.k
    coh = np.flatnonzero(D["lab"] & ~np.isnan(D["I"][:, 0]))
    assert np.isfinite(D["fold"][coh]).all()
    Cr = D["C"][coh]
    keep = np.nanstd(Cr, 0) > 0
    Cr, cnames = Cr[:, keep], [c for c, kp in zip(D["names"], keep) if kp]
    I, R, y, fold = D["I"][coh], D["R"][coh], D["y"][coh], D["fold"][coh]
    has_r = ~np.isnan(R[:, 0])
    nc = len(coh)
    log(f"A. cohort {nc} patients with an image ({has_r.sum()} also RNAseq); clinical view {Cr.shape[1]} variables")

    models = args.models.split(",")
    folds = [int(x) for x in args.folds.split(",")]
    only = tuple(s for s in args.only.split(";") if s)
    res = Parallel(n_jobs=len(folds))(delayed(run_fold)(f, Cr, I, R, y, fold, has_r, k, models, args.n_iter, args.h, only) for f in folds)
    pred = {}
    sel = {"ALDSL": [[], [], []], "ALDSL_C": [[], [], []], "corr": [[], [], []]}
    lam_f = []
    for f, (tem, pf, sf, lam, msg) in zip(folds, res):
        for key, p in pf.items():
            pred.setdefault(key, np.full(nc, np.nan))[tem] = p
        for nm in sel:
            for v in range(3):
                sel[nm][v].append(sf[nm][v])
        lam_f.append(lam)
        log(f"fold {f}: {tem.sum()} held-out; {msg}")

    done = np.isfinite(next(iter(pred.values())))
    # the random draws: per patient mean absolute error over the draws
    err = {}
    for (name, mo), p in pred.items():
        base = name.split(" #")[0]
        err.setdefault((base, mo), []).append(np.abs(p - y))
    err = {key: np.mean(v, 0) for key, v in err.items()}
    groups = {"all with image": done, "image + RNAseq": done & has_r, "image only": done & ~has_r}
    for mo in models:
        out(f"\nA. MAE of the 2026 label, downstream {mo.upper()}, pooled over the held-out folds; gain = MAE(clinical) - MAE(set), 95% interval")
        out(f"{'feature set':46s} " + " ".join(f"{g + ' (' + str(s.sum()) + ')':>22s}" for g, s in groups.items()) + "   gain (all) [95%]          gain (RNAseq)")
        ref = err[("clinical (all)", mo)]
        for (name, m2), e in err.items():
            if m2 != mo:
                continue
            gall, grna = ref[done] - e[done], ref[done & has_r] - e[done & has_r]
            lo, hi = boot(gall)
            lo2, hi2 = boot(grna)
            out(f"{name:46s} " + " ".join(f"{e[s].mean():22.3f}" for s in groups.values()) +
                f"   {gall.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   {grna.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
    out(f"\nA. reference: the CV predictions of the same patients (each from the fold model that held the patient out)")
    for nm, key in (("GRU (sequence model, hs_e2e.py)", "gru"), ("XGBoost tuned (xgb_e2e.py)", "xgb")):
        e = np.abs(D[key][coh] - y)
        out(f"{nm:46s} " + " ".join(f"{e[s].mean():22.3f}" for s in groups.values()))
    for mo in models:
        a_, b_ = (f"clinical + ALDSL latent H ({args.h})", mo), ("clinical + 16 PCs each (GRU input)", mo)
        if a_ not in err or b_ not in err:
            continue
        for nm, s in groups.items():
            dlt = err[a_][s] - err[b_][s]
            lo, hi = boot(dlt)
            out(f"   {mo.upper()}, {nm}: MAE(clinical + H) - MAE(clinical + 16 PCs each) = {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for mo in models:
        best = min((n for (n, m2) in err if m2 == mo), key=lambda n: err[(n, mo)][done].mean())
        for nm, key in (("GRU", "gru"), ("XGBoost", "xgb")):
            dlt = err[(best, mo)][done] - np.abs(D[key][coh] - y)[done]
            lo, hi = boot(dlt)
            out(f"   best {mo.upper()} set ({best}) - {nm}: {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out(f"\nC1. stability of the top {k} over the folds (mean pairwise Jaccard; random sets of k of d: about k / (2d - k))")
    dims = (Cr.shape[1], 256, 128)
    for nm, s in sel.items():
        if len(s[0]) > 1:
            out(f"   {nm:8s} " + ", ".join(f"{VIEWS[v]} {jaccard_mean(s[v]):.2f}" for v in range(3)) +
                "   (random: " + ", ".join(f"{k / (2 * dv - k):.2f}" for dv in dims) + ")")
    out("   overlap ALDSL / supervised |corr| top sets, mean over folds: " +
        ", ".join(f"{VIEWS[v]} {np.mean([len(set(a) & set(b)) for a, b in zip(sel['ALDSL'][v], sel['corr'][v])]):.1f} of {k}" for v in range(3)))
    out("   overlap ALDSL / ALDSL_C: " +
        ", ".join(f"{VIEWS[v]} {np.mean([len(set(a) & set(b)) for a, b in zip(sel['ALDSL'][v], sel['ALDSL_C'][v])]):.1f} of {k}" for v in range(3)))
    out("   view weights lambda (clinical / image / RNAseq), mean over folds: " + " / ".join(f"{x:.3f}" for x in np.mean(lam_f, 0)))
    # one fit on the whole cohort (no labels): the clinical variables it selects
    Cs, Is = standardize(Cr, np.ones(nc, bool)), standardize(I, np.ones(nc, bool))
    Rs = standardize(R, has_r)
    m = ALDSL(n_iter=args.n_iter).fit([Cs, Is, Rs[has_r]], [np.arange(nc), np.arange(nc), np.flatnonzero(has_r)], nc)
    rc = abs_corr(Cs, y)
    out(f"   whole cohort fit (no labels): top {k} clinical variables by ALDSL (|corr| with the 2026 label, rank of 1..{Cr.shape[1]} by |corr|):")
    rank_corr = np.argsort(np.argsort(-rc)) + 1
    for j in m.rank(0)[:k]:
        out(f"      {cnames[j]:28s} |corr| {rc[j]:.2f} (rank {rank_corr[j]})")
    out(f"   top {k} by |corr| for comparison: " + ", ".join(cnames[j] for j in np.argsort(-rc)[:k]))
    out("   |corr| of the latent H columns with the 2026 label: " + ", ".join(f"{x:.2f}" for x in abs_corr(m.H, y)))
    return m, (Cs, Is, Rs, has_r, coh)


def task_b(D, args, m_cut, cut):
    k = args.k
    S = responder_starts()
    pos = {p: i for i, p in enumerate(D["pid"])}
    yb = (S.ratio < 0.5).to_numpy()
    pat = np.array([pos[p] for p in S.patient_id])
    ada = (S.drug == "Adalimumab").to_numpy()
    out(f"\nB. responder starts: {len(S)} ({yb.sum()} good); Adalimumab {ada.sum()} ({yb[ada].sum()} good)")
    _, _, _, has_r, coh = cut
    Rr = D["R"][coh[has_r]]
    Rb = (D["R"][pat] - Rr.mean(0)) / Rr.std(0)  # the scaling of the cutoff fit
    assert np.isfinite(Rb).all()
    # the decision-step fit: clinical at the day before the start, image if taken before, RNAseq (600 patients, no information after the start)
    dC, dI, dR = D["dec_C"], D["dec_I"], D["dec_R"]
    keep = np.nanstd(dC, 0) > 0
    dC = standardize(dC[:, keep], np.ones(len(dC), bool))
    hi_ = ~np.isnan(dI[:, 0])
    dIs = standardize(dI, hi_)
    dRs = standardize(dR, np.ones(len(dR), bool))
    nd = len(dC)
    md = ALDSL(n_iter=args.n_iter).fit([dC, dIs[hi_], dRs], [np.arange(nd), np.flatnonzero(hi_), np.arange(nd)], nd)
    log(f"decision-step fit: {nd} RNAseq patients, {hi_.sum()} with an image before the start; lambda " + " / ".join(f"{x:.3f}" for x in md.lam))
    dpos = {p: i for i, p in enumerate(D["dec_pat"])}
    assert all(p in dpos for p in pat)
    Rd = dRs[[dpos[p] for p in pat]]
    Hall = md.latent([None, None, dRs], [None, None, np.arange(nd)], nd, only=[2])  # RNAseq part of H only
    Hd = standardize(Hall, np.ones(nd, bool))[[dpos[p] for p in pat]]
    # PCA as rna_profiles.py: raw embedding, all 600 samples
    pca = PCA(20).fit(D["rna_emb"])
    Rraw = D["R"][pat]
    Z = pca.transform(Rraw) / np.sqrt(pca.explained_variance_[None, :])
    clin = ["B", "n_pre", "last_pre", "mean_all_pre", "age", "sex", "bmi", "hurley", "delay", "line"]
    Cc = S[clin].astype(float)
    Cc = ((Cc.fillna(Cc.median()) - Cc.mean()) / Cc.std()).to_numpy()

    def oof_auc(A, ybb, reps=5):
        a = []
        for rep in range(reps):
            p = np.zeros(len(ybb))
            for tr, te in KFold(5, shuffle=True, random_state=rep).split(A):
                p[te] = LogisticRegression(C=1.0, max_iter=2000).fit(A[tr], ybb[tr]).predict_proba(A[te])[:, 1]
            a.append(roc_auc_score(ybb, p))
        return float(np.mean(a))

    rk_cut, rk_dec = m_cut.rank(2), md.rank(2)
    rng = np.random.default_rng(1)
    rand = {kk: [rng.choice(128, kk, replace=False) for _ in range(20)] for kk in (1, 3, k)}
    out(f"   out-of-fold AUC of a good response (logistic, 5 x 5-fold); random dims: mean over 20 draws [range], and the ALDSL set's place among them")
    for name, s in (("all starts", np.ones(len(S), bool)), ("Adalimumab", ada)):
        out(f"   {name} (n={s.sum()}, good {yb[s].sum()})")
        out(f"      {'clinical (10 inputs of rna_profiles.py)':52s} {oof_auc(Cc[s], yb[s]):.3f}")
        out(f"      {'all 128 RNAseq dims':52s} {oof_auc(Rb[s], yb[s]):.3f}")
        for kk in (1, 3, k):
            ra = [oof_auc(Rb[s][:, j], yb[s], reps=2) for j in rand[kk]]
            a_cut, a_dec = oof_auc(Rb[s][:, rk_cut[:kk]], yb[s]), oof_auc(Rd[s][:, rk_dec[:kk]], yb[s])
            out(f"      top {kk:2d}: PCs {oof_auc(Z[s][:, :kk], yb[s]):.3f} | ALDSL cutoff fit {a_cut:.3f} (above {np.mean(np.array(ra) < a_cut):.0%} of random) | "
                f"ALDSL decision fit {a_dec:.3f} (above {np.mean(np.array(ra) < a_dec):.0%}) | random {np.mean(ra):.3f} [{min(ra):.3f}, {max(ra):.3f}]")
        out(f"      {'ALDSL decision fit, RNAseq part of H (' + str(md.h) + ')':52s} {oof_auc(Hd[s], yb[s]):.3f}")
        out(f"      {'clinical + PC1':52s} {oof_auc(np.column_stack([Cc[s], Z[s][:, :1]]), yb[s]):.3f}")
        out(f"      {'clinical + ALDSL decision fit top 1':52s} {oof_auc(np.column_stack([Cc[s], Rd[s][:, rk_dec[:1]]]), yb[s]):.3f}")
        out(f"      {'clinical + ALDSL decision fit, RNAseq part of H':52s} {oof_auc(np.column_stack([Cc[s], Hd[s]]), yb[s]):.3f}")
    ld = np.abs(pca.components_[0])
    top_pc1 = set(np.argsort(-ld)[:k])
    out(f"   overlap with the {k} dims of largest |PC1 loading|: ALDSL cutoff fit {len(top_pc1 & set(rk_cut[:k]))}, decision fit "
        f"{len(top_pc1 & set(rk_dec[:k]))} (random about {k * k / 128:.1f})")
    out(f"   corr(|PC1 loading|, row norm of P): cutoff fit {np.corrcoef(ld, np.linalg.norm(m_cut.P[2], axis=1))[0, 1]:+.2f}, decision fit "
        f"{np.corrcoef(ld, np.linalg.norm(md.P[2], axis=1))[0, 1]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="selected variables per view (the paper reports the top 10)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--models", default="svr,hgb")
    ap.add_argument("--n_iter", type=int, default=60)
    ap.add_argument("--h", type=int, default=5, help="dimension of the latent H (the reference code: 5)")
    ap.add_argument("--only", default="", help="task A: only the feature sets whose names start with one of these (';'-separated)")
    ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--out", default="runs/aldsl_hs.txt")
    args = ap.parse_args()
    D = load()
    log("data loaded")
    m_cut, cut = task_a(D, args)
    if not args.skip_b:
        task_b(D, args, m_cut, cut)
    log("done")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
