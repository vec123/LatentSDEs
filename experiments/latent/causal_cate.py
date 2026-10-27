"""
Who should get Bimekizumab and who Secukinumab? Heterogeneous effects on line 2 starts (Adalimumab -> Bimekizumab 1,063 / -> Secukinumab 1,039
patients in the train folder; the first newer drug is chosen without a visible rule, propensity AUC 0.505). Default env, CPU, about 2 minutes.

Unit and outcome as in causal_bime_secu.py: line 2 starts with an observed outcome; outcome higher = better: minus the ratio (post / pre, capped at 3)
and the good responder flag (ratio < 0.5). Inputs: the step features of the decision step.
Method (every number below uses only predictions for patients that the model did not see; 5 repeats of 5-fold cross-fitting by patient):
  - T-learner: boosted trees per drug; predicted benefit of Bimekizumab over Secukinumab tau(x) = m_Bime(x) - m_Secu(x);
  - effect inside groups of predicted tau (quartiles) with the doubly robust (AIPW) score of each patient; a real heterogeneity shows as a
    steady rise of the effect from the first to the last quartile;
  - policy value: treat with the drug that tau favours, against all Bimekizumab, all Secukinumab and a coin flip; AIPW estimate of the mean outcome
    under each policy, differences with a 95% bootstrap interval over patients.
Observational data: a group effect is a hypothesis. The propensity is taken as the share of Bimekizumab (the choice is unrelated to the inputs).

    python -I experiments/latent/causal_cate.py
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

rng = np.random.default_rng(0)
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
names = [str(n) for n in d["feature_names"]]
X0, M, kind, day, off = d["X"].astype(np.float64), d["M"] > 0, d["kind"], d["day"].astype(np.int64), d["offsets"]
prow = np.repeat(np.arange(len(off) - 1), np.diff(off))
own = names.index("own_score")
has = (kind == 0) & M[:, own]
key = {(int(p), int(s)): (int(k), int(l)) for p, s, k, l in zip(d["bio_patient"], d["bio_start"].astype(np.int64), d["bio_drug"], d["bio_line"])}
drop = {"is_decision", "is_query", "is_final", "visit_month", "own_score"}
cols = [j for j, n in enumerate(names) if n not in drop and not n.startswith("bio_active_") and n not in ("days_on_bio", "cur_bio_line")]
rows, A, ratio = [], [], []
for r in np.flatnonzero(kind == 3):
    k = key.get((int(prow[r]), int(day[r]) + 1))
    if k is None or k[0] not in (1, 2) or k[1] != 2:
        continue
    i, s = prow[r], day[r] + 1
    sel = np.arange(off[i], off[i + 1])[has[off[i]:off[i + 1]]]
    dd, sv = day[sel] - s, X0[sel, own]
    pre, post = sv[(dd >= -180) & (dd < 0)], sv[(dd >= 90) & (dd <= 270)]
    if len(pre) and len(post) and pre.mean() >= 3:
        rows.append(r); A.append(int(k[0] == 1)); ratio.append(post.mean() / pre.mean())
rows, A, ratio = np.array(rows), np.array(A), np.minimum(np.array(ratio), 3)
pat = prow[rows]
X = np.where(M[rows][:, cols], X0[rows][:, cols], np.nan)  # trees take missing values
cn = [names[j] for j in cols]
print(f"{len(rows)} line 2 starts with an outcome; Bimekizumab {A.sum()}, Secukinumab {(1 - A).sum()}")
e = A.mean()
u, inv = np.unique(pat, return_inverse=True)
groups = [np.flatnonzero(inv == k) for k in range(len(u))]
P = dict(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, n_jobs=8, verbosity=0)


def crossfit(y, reps=5):
    """out-of-fold tau, m1, m0 (mean over repeats)"""
    T, M1, M0 = (np.zeros((reps, len(y))) for _ in range(3))
    for rep in range(reps):
        g = rng.permutation(len(u))[inv]
        for tr, te in GroupKFold(5).split(X, y, g):
            for a, out in ((1, M1), (0, M0)):
                t2 = tr[A[tr] == a]
                out[rep, te] = xgb.XGBRegressor(**P).fit(X[t2], y[t2]).predict(X[te])
    m1, m0 = M1.mean(0), M0.mean(0)
    return m1 - m0, m1, m0


def boot(fn, n=500):
    return np.percentile([fn(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(n)], [2.5, 97.5])


for title, y in (("outcome: minus the ratio post / pre (higher = better)", -ratio), ("outcome: good responder (ratio < 0.5)", (ratio < 0.5).astype(float))):
    tau, m1, m0 = crossfit(y)
    G = m1 - m0 + A * (y - m1) / e - (1 - A) * (y - m0) / (1 - e)  # doubly robust effect score of each patient (Bime - Secu)
    print(f"\n{title}\n   average effect Bime - Secu: {G.mean():+.4f} [{boot(lambda ix: G[ix].mean())[0]:+.4f}, {boot(lambda ix: G[ix].mean())[1]:+.4f}]")
    print(f"   spread of the out-of-fold predicted tau: sd {tau.std():.4f}; corr(tau, patient effect score) {np.corrcoef(tau, G)[0, 1]:+.3f}")
    q = np.quantile(tau, [0.25, 0.5, 0.75])
    qi = np.searchsorted(q, tau)
    print("   effect by quartile of the predicted tau (low = Secukinumab favoured, high = Bimekizumab favoured):")
    for k in range(4):
        s = qi == k
        lo, hi = boot(lambda ix: G[ix][s[ix]].mean() if s[ix].any() else np.nan)
        print(f"      quartile {k + 1}: n={s.sum():4d}  mean predicted tau {tau[s].mean():+.4f}  effect {G[s].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    top_minus_bottom = lambda ix: G[ix][qi[ix] == 3].mean() - G[ix][qi[ix] == 0].mean()
    lo, hi = boot(top_minus_bottom)
    print(f"   quartile 4 minus quartile 1: {top_minus_bottom(np.arange(len(y))):+.4f} [{lo:+.4f}, {hi:+.4f}]  (> 0 and excluding 0: the model finds real differences)")
    pol = (tau > 0).astype(float)  # 1: give Bimekizumab
    def val(p, ix):
        a, yy = A[ix], y[ix]
        b = m1[ix] + a * (yy - m1[ix]) / e
        s_ = m0[ix] + (1 - a) * (yy - m0[ix]) / (1 - e)
        return (p[ix] * b + (1 - p[ix]) * s_).mean()
    allb, alls = np.ones(len(y)), np.zeros(len(y))
    ix0 = np.arange(len(y))
    print(f"   policy 'give the drug that tau favours': {pol.mean():.2f} of the patients get Bimekizumab")
    for nm, ref in (("all Bimekizumab", allb), ("all Secukinumab", alls), ("coin flip", np.full(len(y), 0.5))):
        f = lambda ix: val(pol, ix) - val(ref, ix)
        lo, hi = boot(f)
        print(f"      value minus {nm:16s} {f(ix0):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    j = np.argsort(-np.abs([np.corrcoef(np.nan_to_num(X[:, c], nan=np.nanmedian(X[:, c])), tau)[0, 1] if np.nanstd(X[:, c]) > 0 else 0 for c in range(X.shape[1])]))[:6]
    print("   inputs most correlated with tau: " + ", ".join(f"{cn[c]} {np.corrcoef(np.nan_to_num(X[:, c], nan=np.nanmedian(X[:, c])), tau)[0, 1]:+.2f}" for c in j))
