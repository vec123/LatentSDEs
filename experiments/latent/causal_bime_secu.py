"""
Bimekizumab against Secukinumab: an observational effect estimate (default env, CPU, about 1 minute).
Why this pair: Adalimumab is always the first biologic and the newer drugs always follow one, so only these two drugs are chosen
independently of the observed inputs (assignment.py: AUC of the choice from EHR + calendar day 0.52).
Unit: each Bimekizumab / Secukinumab start of the train folder, inputs = the 65 step features of the decision step (the day before the start).
Outcome: ratio = mean score in days 90..270 after the start / mean score in the 180 days before (pre mean >= 3, a score in both windows,
as in rna_response.py); good responder = ratio < 0.5; also the change of the mean score (post - pre). Intention to treat: the drug that was
started counts, even if it stopped later.
Checks before the estimate: balance of the inputs (standardized differences), overlap of the propensity score, whether the outcome is observed
equally often under both drugs (selection), and a negative-control outcome (the mean score in days -360..-180 before the start, which the
start cannot change).
Estimators: difference in means; regression adjustment (ridge per arm); AIPW (cross-fitted logistic propensity, ridge outcome models).
95% intervals from a bootstrap over patients (300 draws; the models are fitted once, the cross-fitted nuisance values are resampled). The intervals of the regression-adjusted column ignore the uncertainty of the model fit and are too narrow: read the difference in means and AIPW.
A causal reading needs: no hidden confounder, overlap, and one version of each treatment. None of these is proven by the checks.

    python -I experiments/latent/causal_bime_secu.py --line2     # line 2 only: the valid comparison (see the note in the handoff)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

rng = np.random.default_rng(0)
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
names = [str(n) for n in d["feature_names"]]
X0, M, kind, day, off = d["X"].astype(np.float64), d["M"] > 0, d["kind"], d["day"].astype(np.int64), d["offsets"]
prow = np.repeat(np.arange(len(off) - 1), np.diff(off))
own = names.index("own_score")
has = (kind == 0) & M[:, own]
key = {(int(p), int(s)): int(k) for p, s, k in zip(d["bio_patient"], d["bio_start"].astype(np.int64), d["bio_drug"])}
drop = {"is_decision", "is_query", "is_final", "visit_month", "own_score"}
cols = [j for j, n in enumerate(names) if n not in drop and not n.startswith("bio_active_") and n not in ("days_on_bio", "cur_bio_line")]
rows, drug, ratio, delta, older, pre_m = [], [], [], [], [], []
for r in np.flatnonzero(kind == 3):
    k = key.get((int(prow[r]), int(day[r]) + 1))
    if k not in (1, 2):
        continue
    s = day[r] + 1
    i = prow[r]
    sel = np.arange(off[i], off[i + 1])[has[off[i]:off[i + 1]]]
    dd, sv = day[sel] - s, X0[sel, own]
    pre, post, old = sv[(dd >= -180) & (dd < 0)], sv[(dd >= 90) & (dd <= 270)], sv[(dd >= -360) & (dd < -180)]
    rows.append(r); drug.append(k == 1)
    pre_m.append(pre.mean() if len(pre) else np.nan)
    ok = len(pre) and len(post) and pre.mean() >= 3
    ratio.append(post.mean() / pre.mean() if ok else np.nan)
    delta.append(post.mean() - pre.mean() if ok else np.nan)
    older.append(old.mean() if len(old) else np.nan)
rows, A = np.array(rows), np.array(drug, int)  # A = 1: Bimekizumab
ratio, delta, older, pre_m = map(np.array, (ratio, delta, older, pre_m))
import sys
if "--line2" in sys.argv:  # line 3 starts take the drug that the patient did not have in line 2 (no choice): keep line 2 only
    k2 = X0[rows, names.index("max_bio_line")] == 1
    rows, A, ratio, delta, older, pre_m = rows[k2], A[k2], ratio[k2], delta[k2], older[k2], pre_m[k2]
    print("line 2 starts only (prior line 1)")
pat = prow[rows]
print(f"{len(rows)} starts: Bimekizumab {A.sum()}, Secukinumab {(1 - A).sum()}")
obs = np.isfinite(ratio)
print(f"\nSelection: outcome observed (both windows, pre mean >= 3): Bimekizumab {obs[A == 1].mean():.3f}, Secukinumab {obs[A == 0].mean():.3f}")
for nm, s in (("any score in days 90..270", None),):
    pass
X = np.where(M[rows][:, cols], X0[rows][:, cols], np.nan)
X = np.column_stack([X, np.isnan(X).any(0) * 0 + 0][:1]) if False else X
med = np.nanmedian(X, 0)
miss = np.isnan(X)
X = np.where(miss, med, X)
keep = (X.std(0) > 0)
X = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
cn = [names[j] for j in np.array(cols)[keep]]
print(f"{X.shape[1]} inputs")

# ---- balance and overlap on the starts with an observed outcome
s = obs
w1 = X[s & (A == 1)].mean(0) - X[s & (A == 0)].mean(0)
print(f"\nBalance (starts with an outcome, n={s.sum()}): the largest standardized differences of the inputs between the drugs")
for j in np.argsort(-np.abs(w1))[:6]:
    print(f"   {cn[j]:26s} {w1[j]:+.2f}")
print(f"   share of inputs with |difference| > 0.1: {np.mean(np.abs(w1) > 0.1):.2f}")
Xs, As, ys_pat = X[s], A[s], pat[s]
Y = {"ratio (capped at 3)": np.minimum(ratio[s], 3), "good responder (ratio < 0.5)": (ratio[s] < 0.5).astype(float), "change of the mean score": delta[s]}
nc = older[s]
fold = list(GroupKFold(5).split(Xs, As, ys_pat))
e = np.zeros(len(As))
for tr, te in fold:
    e[te] = LogisticRegression(C=0.05, max_iter=500).fit(Xs[tr], As[tr]).predict_proba(Xs[te])[:, 1]
from sklearn.metrics import roc_auc_score
print(f"cross-fitted propensity: AUC {roc_auc_score(As, e):.3f}; range {e.min():.2f}..{e.max():.2f}; share outside 0.2..0.8: {np.mean((e < 0.2) | (e > 0.8)):.3f}")
ec = np.clip(e, 0.1, 0.9)


def nuis(y):
    m1, m0 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in fold:
        for a, m in ((1, m1), (0, m0)):
            t2 = tr[As[tr] == a]
            m[te] = Ridge(alpha=300.0).fit(Xs[t2], y[t2]).predict(Xs[te])
    return m1, m0


def estimates(idx, y, m1, m0):
    a, yy, ee = As[idx], y[idx], ec[idx]
    dm = yy[a == 1].mean() - yy[a == 0].mean()
    ra = (m1[idx] - m0[idx]).mean()
    psi = m1[idx] - m0[idx] + a * (yy - m1[idx]) / ee - (1 - a) * (yy - m0[idx]) / (1 - ee)
    return dm, ra, psi.mean()


u, inv = np.unique(ys_pat, return_inverse=True)
groups = [np.flatnonzero(inv == k) for k in range(len(u))]
print("\nEffect of Bimekizumab minus Secukinumab (starts with an observed outcome; 95% interval, bootstrap over patients)")
print(f"   {'outcome':30s} {'difference in means':>24s} {'regression adjusted':>24s} {'AIPW':>24s}")
for nm, y in list(Y.items()) + [("negative control: mean score -360..-180 d", nc)]:
    okk = np.isfinite(y)
    m1, m0 = nuis(np.where(okk, y, 0))
    base = np.flatnonzero(okk)
    pt = estimates(base, np.where(okk, y, 0), m1, m0)
    bs = np.array([estimates(np.intersect1d(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]), base), np.where(okk, y, 0), m1, m0)
                   for _ in range(300)])
    f = lambda k: f"{pt[k]:+.3f} [{np.percentile(bs[:, k], 2.5):+.3f}, {np.percentile(bs[:, k], 97.5):+.3f}]"
    print(f"   {nm:30s} {f(0):>24s} {f(1):>24s} {f(2):>24s}")
print(f"\nmean outcomes: ratio Bimekizumab {np.minimum(ratio[s], 3)[As == 1].mean():.3f}, Secukinumab {np.minimum(ratio[s], 3)[As == 0].mean():.3f}; "
      f"good responders {np.mean(ratio[s][As == 1] < 0.5):.3f} / {np.mean(ratio[s][As == 0] < 0.5):.3f}")
# ---- subgroups (AIPW, good responder)
y = Y["good responder (ratio < 0.5)"]
m1, m0 = nuis(y)
print("\nSubgroups (AIPW, good-responder share; wide intervals expected)")
line = X0[rows][s, names.index("max_bio_line")]
hur = X0[rows][s, names.index("hurley_stage_baseline")]
for nm, g in (("prior line 1 (this is line 2)", line == 1), ("prior line >= 2 (this is line 3)", line >= 2), ("Hurley stage <= 2", hur <= 2), ("Hurley stage 3", hur >= 3)):
    base = np.flatnonzero(g)
    if len(base) < 60 or As[base].min() == As[base].max():
        continue
    pt = estimates(base, y, m1, m0)[2]
    bs = [estimates(np.intersect1d(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]), base), y, m1, m0)[2] for _ in range(200)]
    print(f"   {nm:36s} n={len(base):4d} (Bime {As[base].sum()})  {pt:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
r_ok = obs & (np.isfinite(ratio))
print(f"\nstarts with RNAseq and an observed outcome: {int((d['rna_row'][rows] >= 0)[r_ok].sum())} (too few for an RNAseq-specific effect)")
