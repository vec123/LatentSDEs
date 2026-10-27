"""
Which patient features change the effect of Bimekizumab against Secukinumab? Pre-specified effect modifiers on line 2 starts (default env, CPU,
< 1 min). Line 2 is the only near-random choice (assignment.py, causal_bime_secu.py --line2); at line 3 the drug is the one not used before.

Effect scores: the doubly robust (AIPW) score of each start, Gamma_i = m1 - m0 + A (y - m1) / e - (1 - A) (y - m0) / (1 - e), with outcome models
(ridge per drug on the EHR features of the decision step) and the propensity (logistic) cross-fitted over 5 folds by patient. Gamma estimates the
effect of Bimekizumab minus Secukinumab for that start; outcome "higher = better" = minus the ratio post / pre (capped at 3).
Modifier test: the best linear projection Gamma ~ 1 + z (z = one modifier, standardized), slope with a robust (HC1) standard error; slope > 0 means
that Bimekizumab gets better relative to Secukinumab as z grows. Holm correction over the modifiers. MDE = the slope that the test finds with 80%
power (2.8 standard errors).
Modifiers (fixed before looking): the response to Adalimumab in line 1 (ratio post / pre; post window cut at the line 2 start), the months on
Adalimumab before the switch, Hurley stage, BMI, current smoking, sex, age, disease duration, recent mean score, surgeries; RNAseq PC1-PC3 (sample on
the start day) and image PC1-PC3 (latest image <= 365 days before), where present.
Outcomes: O1 = days 90..270 after the start, pre mean >= 3 (as before); O2 = days 30..270, pre mean >= 1 (more starts with RNAseq / image).

    python -I experiments/latent/causal_modifiers.py
"""
from math import erf, sqrt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

MONTH = 30.44
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
names = [str(n) for n in d["feature_names"]]
X0, M, kind, day, off = d["X"].astype(np.float64), d["M"] > 0, d["kind"], d["day"].astype(np.int64), d["offsets"]
prow = np.repeat(np.arange(len(off) - 1), np.diff(off))
own = names.index("own_score")
has = (kind == 0) & M[:, own]
nm = names.index
bp, bs, bd, bl = d["bio_patient"], d["bio_start"].astype(np.int64), d["bio_drug"], d["bio_line"]
key = {(int(p), int(s)): (int(k), int(l)) for p, s, k, l in zip(bp, bs, bd, bl)}
line1 = {}
for p, s, l in zip(bp, bs, bl):
    if l == 1 and (int(p) not in line1 or s < line1[int(p)]):
        line1[int(p)] = int(s)
drop = {"is_decision", "is_query", "is_final", "visit_month", "own_score"}
cols = [j for j, n in enumerate(names) if n not in drop and not n.startswith("bio_active_") and n not in ("days_on_bio", "cur_bio_line")]
rpca, ipca = PCA(3).fit(d["rna_emb"]), PCA(3).fit(d["img_emb"])


def scores(i):
    sel = np.arange(off[i], off[i + 1])[has[off[i]:off[i + 1]]]
    return day[sel], X0[sel, own]


def ratio_at(sd, sv, s, lo, hi, minpre, stop=None):
    dd = sd - s
    hi_ = hi if stop is None else min(hi, stop - s - 1)
    pre, post = sv[(dd >= -180) & (dd < 0)], sv[(dd >= lo) & (dd <= hi_)]
    return post.mean() / pre.mean() if len(pre) and len(post) and pre.mean() >= minpre and hi_ >= lo else np.nan


def build(lo, hi, minpre):
    R = []
    for r in np.flatnonzero(kind == 3):
        k = key.get((int(prow[r]), int(day[r]) + 1))
        if k is None or k[0] not in (1, 2) or k[1] != 2:
            continue
        i, s = int(prow[r]), int(day[r]) + 1
        sd, sv = scores(i)
        y = ratio_at(sd, sv, s, lo, hi, minpre)
        if not np.isfinite(y):
            continue
        s1 = line1.get(i)
        ada = ratio_at(sd, sv, s1, 90, 270, 3, stop=s) if s1 is not None else np.nan
        f = lambda c: X0[r, nm(c)] if M[r, nm(c)] else np.nan
        rna = rpca.transform(d["rna_emb"][[d["rna_row"][r]]])[0] if d["rna_row"][r] >= 0 else np.full(3, np.nan)
        img_ok = d["img_row"][r] >= 0 and M[r, nm("img_days_since")] and X0[r, nm("img_days_since")] <= 365
        img = ipca.transform(d["img_emb"][[d["img_row"][r]]])[0] if img_ok else np.full(3, np.nan)
        mods = {"Adalimumab response (ratio, line 1)": ada, "months on Adalimumab": (s - s1) / MONTH if s1 is not None else np.nan,
                "Hurley stage": f("hurley_stage_baseline"), "BMI": f("bmi_at_baseline"), "current smoker": f("smoking_status"), "sex": f("sex"),
                "age": f("age_at_visit"), "disease duration": f("days_since_diag"), "recent mean score (6 m)": f("mean_6m"), "surgeries": f("n_surg"),
                **{f"RNAseq PC{q + 1}": rna[q] for q in range(3)}, **{f"image PC{q + 1}": img[q] for q in range(3)}}
        R.append((r, i, int(k[0] == 1), min(y, 3.0), mods))
    return R


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def aipw(R):
    """the AIPW score of each start of R (Bimekizumab minus Secukinumab on minus the ratio), with the treatment and the outcome"""
    rows = np.array([x[0] for x in R]); pat = np.array([x[1] for x in R]); A = np.array([x[2] for x in R]); y = -np.array([x[3] for x in R])
    X = np.where(M[rows][:, cols], X0[rows][:, cols], np.nan)
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    X = X[:, X.std(0) > 0]
    X = (X - X.mean(0)) / X.std(0)
    m1, m0, e = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, pat):
        e[te] = LogisticRegression(C=0.05, max_iter=500).fit(X[tr], A[tr]).predict_proba(X[te])[:, 1]
        for a, out in ((1, m1), (0, m0)):
            t2 = tr[A[tr] == a]
            out[te] = Ridge(alpha=300.0).fit(X[t2], y[t2]).predict(X[te])
    e = np.clip(e, 0.1, 0.9)
    return m1 - m0 + A * (y - m1) / e - (1 - A) * (y - m0) / (1 - e), A, y


def analyse(title, R):
    G, A, y = aipw(R)
    print(f"\n{title}: {len(y)} line 2 starts (Bimekizumab {A.sum()}); average effect Bime - Secu on minus the ratio: {G.mean():+.3f} "
          f"(SE {G.std() / np.sqrt(len(G)):.3f})")
    print(f"   {'modifier':36s} {'n':>5s} {'slope per SD':>13s} {'95% interval':>18s} {'p':>7s} {'Holm p':>7s} {'MDE':>6s}")
    out = []
    for name in R[0][4]:
        z = np.array([x[4][name] for x in R], float)
        ok = np.isfinite(z)
        if ok.sum() < 20 or np.std(z[ok]) == 0:
            out.append((name, ok.sum(), np.nan, np.nan, np.nan))
            continue
        zz = (z[ok] - z[ok].mean()) / z[ok].std()
        D = np.column_stack([np.ones(ok.sum()), zz])
        Ginv = np.linalg.inv(D.T @ D)
        b = Ginv @ D.T @ G[ok]
        res = G[ok] - D @ b
        V = Ginv @ (D.T * res ** 2) @ D @ Ginv * ok.sum() / (ok.sum() - 2)
        se = np.sqrt(V[1, 1])
        out.append((name, ok.sum(), b[1], se, 2 * (1 - phi(abs(b[1] / se)))))
    ps = np.array([o[4] for o in out])
    order = np.argsort(np.where(np.isnan(ps), 9, ps))
    m = np.isfinite(ps).sum()
    holm = np.full(len(ps), np.nan)
    run = 0.0
    for rank, j in enumerate(order[:m]):
        run = max(run, min(1.0, (m - rank) * ps[j]))
        holm[j] = run
    for (name, n, b, se, p), h in zip(out, holm):
        if np.isnan(b):
            print(f"   {name:36s} {n:5d}   too few")
            continue
        print(f"   {name:36s} {n:5d} {b:+13.3f} {f'[{b - 1.96 * se:+.3f}, {b + 1.96 * se:+.3f}]':>18s} {p:7.3f} {h:7.3f} {2.8 * se:6.3f}")
    # the two Adalimumab modifiers in thirds (effect = mean Gamma)
    for name in ("Adalimumab response (ratio, line 1)", "months on Adalimumab"):
        z = np.array([x[4][name] for x in R], float)
        ok = np.isfinite(z)
        q = np.quantile(z[ok], [1 / 3, 2 / 3])
        parts = []
        for k, (lo_, hi_) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
            s = ok & (z > lo_) & (z <= hi_)
            parts.append(f"third {k + 1} ({z[s].min():.2f}..{z[s].max():.2f}, n={s.sum()}): {G[s].mean():+.3f} +- {1.96 * G[s].std() / np.sqrt(s.sum()):.3f}")
        print(f"   {name}: " + "; ".join(parts))


if __name__ == "__main__":
    analyse("O1: days 90..270, pre mean >= 3", build(90, 270, 3))
    analyse("O2: days 30..270, pre mean >= 1", build(30, 270, 1))
