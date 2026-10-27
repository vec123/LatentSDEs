"""
RNAseq profiles and response (CPU, about 1 minute). Same starts and response as rna_response.py: the biologic starts with a baseline
RNAseq sample, ratio = mean score days 90..270 after / mean 180 days before (pre mean >= 3); good responder: ratio < 0.5.
The files hold only the 128-d embedding (no genes), so components are the unit.

A. Profiles: (1) does the embedding of the 600 RNAseq patients form groups (k-means, silhouette)? (2) good against other responders: the
   standardized mean difference of each component, per drug and for all starts, against a permutation null (largest |difference|).
B. Adalimumab: out-of-fold predicted ratio from the embedding (ridge, 5-fold x 5 repeats) -> thirds of the prediction -> good share per
   third; AUC with a permutation p-value; clinical inputs alone and with the embedding.
C. Drug comparison (exploratory, confounded): ridge on the clinical inputs + embedding with a drug term and drug x embedding terms
   (pooled Bimekizumab + Secukinumab = "newer" against Adalimumab). Cross-validated: separate fit per drug group against one shared fit.
   The predicted ratio under each drug for each patient; how often the other drug looks better; the realized ratio when the given drug
   is the predicted best.

    python -I experiments/latent/rna_profiles.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import KFold

TRAIN = "test_data/hs_public_dataset/public/train/"
rng = np.random.default_rng(0)
t = pd.read_csv(TRAIN + "treatments.csv", parse_dates=["start_date"])
v = pd.read_csv(TRAIN + "visits.csv", parse_dates=["visit_date"]).dropna(subset=["ihs4_score"])
demo = pd.read_csv(TRAIN + "demographics.csv").set_index("patient_id")
rm = pd.read_csv(TRAIN + "rnaseq_metadata.csv", parse_dates=["collection_date"])
re_ = pd.read_csv(TRAIN + "rnaseq_embeddings.csv").merge(rm[["rnaseq_sample_id", "patient_id", "collection_date"]], on="rnaseq_sample_id")
emb_cols = [c for c in re_.columns if c.startswith("rnaseq_dim_")]
E_all = re_[emb_cols].to_numpy(float)

b = t[t.treatment_class == "Biologic"].merge(re_[["patient_id", "collection_date"] + emb_cols], on="patient_id")
b = b[b.start_date == b.collection_date].drop_duplicates("patient_id")
sc = {p: g for p, g in v.groupby("patient_id")}
rows = []
for r in b.itertuples():
    g = sc.get(r.patient_id)
    if g is None:
        continue
    d = (g.visit_date - r.start_date).dt.days
    pre, post = g.ihs4_score[(d >= -180) & (d < 0)], g.ihs4_score[(d >= 90) & (d <= 270)]
    if len(pre) == 0 or len(post) == 0 or pre.mean() < 3:
        continue
    prior = g.ihs4_score[d < 0]
    dm = demo.loc[r.patient_id]
    rows.append(dict(patient_id=r.patient_id, ratio=post.mean() / pre.mean(), B=pre.mean(), n_pre=len(pre), last_pre=prior.iloc[-1], mean_all_pre=prior.mean(),
                     age=r.start_date.year - dm.birth_year, sex=float(dm.sex == "Female"), bmi=dm.bmi_at_baseline, hurley=dm.hurley_stage_baseline,
                     delay=dm.diagnostic_delay_years, line=r.treatment_line, drug=r.drug_name, **{c: getattr(r, c) for c in emb_cols}))
D = pd.DataFrame(rows)
y, lab = D.ratio.to_numpy(), D.ratio.to_numpy() < 0.5
drug = D.drug.to_numpy()
print(f"{len(D)} starts; good responders {lab.sum()}; " + ", ".join(f"{k} {int((drug == k).sum())} (good {int(lab[drug == k].sum())})" for k in np.unique(drug)))
clin = ["B", "n_pre", "last_pre", "mean_all_pre", "age", "sex", "bmi", "hurley", "delay", "line"]
C = D[clin].astype(float)
C = ((C.fillna(C.median()) - C.mean()) / C.std()).to_numpy()
pca = PCA(20).fit(E_all)  # unsupervised, on all 600 samples
Z_all = pca.transform(E_all) / np.sqrt(pca.explained_variance_[None, :])
Z = pca.transform(D[emb_cols].to_numpy(float)) / np.sqrt(pca.explained_variance_[None, :])

# ---- A. profiles
print("\nA1. do the 600 RNAseq samples form groups? (k-means on the first 10 components; silhouette: 0 = none, > 0.5 = clear)")
for k in (2, 3, 4):
    lb = KMeans(k, n_init=10, random_state=0).fit_predict(Z_all[:, :10])
    print(f"   k={k}: silhouette {silhouette_score(Z_all[:, :10], lb):.3f}, group sizes {np.bincount(lb).tolist()}")
print(f"   variance of the first 5 components: {np.round(pca.explained_variance_ratio_[:5], 3).tolist()} (20 components: {pca.explained_variance_ratio_.sum():.2f})")
print("\nA2. components that differ between good responders and the others (standardized mean difference; permutation null of the largest)")


def smd(Zs, l):
    return (Zs[l].mean(0) - Zs[~l].mean(0)) / np.sqrt((Zs[l].var(0) * l.sum() + Zs[~l].var(0) * (~l).sum()) / len(l))


for name, s in [("all starts", np.ones(len(D), bool))] + [(k, drug == k) for k in np.unique(drug)]:
    if lab[s].sum() < 3:
        print(f"   {name}: {lab[s].sum()} good responders, no test")
        continue
    d = smd(Z[s], lab[s])
    null = np.array([np.abs(smd(Z[s], rng.permutation(lab[s]))).max() for _ in range(2000)])
    top = np.argsort(-np.abs(d))[:3]
    print(f"   {name:13s} n={s.sum():3d} good={lab[s].sum():2d}  largest |difference| {np.abs(d).max():.2f} (null 95%: {np.percentile(null, 95):.2f}, p={np.mean(null >= np.abs(d).max()):.3f})"
          f"  top components: " + ", ".join(f"PC{j + 1} {d[j]:+.2f}" for j in top))

# ---- B. Adalimumab: groups from the RNAseq prediction
ada = drug == "Adalimumab"


def oof(X, yv, reps=5):
    P = np.zeros((reps, len(yv)))
    for rep in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            P[rep, te] = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(X[tr], yv[tr]).predict(X[te])
    return P


print(f"\nB. Adalimumab starts: {ada.sum()} ({lab[ada].sum()} good). Out-of-fold ridge prediction of the ratio (mean over 5 repeats of 5-fold)")
res = {}
for name, X in {"clinical": C[ada], "RNAseq (20 components)": Z[ada], "clinical + RNAseq": np.column_stack([C[ada], Z[ada]])}.items():
    P = oof(X, y[ada])
    a = [roc_auc_score(lab[ada], -p) for p in P]
    res[name] = P.mean(0)
    print(f"   {name:24s} AUC {np.mean(a):.3f} (+-{np.std(a):.3f})")
P = oof(Z[ada], y[ada])
auc0 = np.mean([roc_auc_score(lab[ada], -p) for p in P])
null = [np.mean([roc_auc_score(l, -p) for p in oof(Z[ada], y[ada][rng.permutation(ada.sum())], reps=1)]) for l in [lab[ada]] for _ in range(100)]
print(f"   permutation null of the RNAseq AUC (100 shuffles of the ratio): mean {np.mean(null):.3f}, 95% {np.percentile(null, 95):.3f}; p = {np.mean(np.array(null) >= auc0):.3f}")
pr = res["RNAseq (20 components)"]
q = np.quantile(pr, [1 / 3, 2 / 3])
th = np.searchsorted(q, pr)
print("   thirds of the out-of-fold predicted ratio (RNAseq only):")
for k, nm in enumerate(["lowest (predicted responders)", "middle", "highest"]):
    s = th == k
    print(f"      {nm:30s} n={s.sum():3d} good share {lab[ada][s].mean():.2f}  median realized ratio {np.median(y[ada][s]):.2f}")
fit = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Z[ada], y[ada])
w = fit.coef_
print("   ridge weights of the RNAseq components (full fit, per standard deviation): " + ", ".join(f"PC{j + 1} {w[j]:+.3f}" for j in np.argsort(-np.abs(w))[:5]))
print("   clinical profile of the good responders against the others (standardized difference):")
dd = smd(C[ada], lab[ada])
print("      " + ", ".join(f"{clin[j]} {dd[j]:+.2f}" for j in np.argsort(-np.abs(dd))[:5]))

# ---- C. drug comparison (exploratory)
new = (drug != "Adalimumab").astype(float)
print(f"\nC. drug comparison, exploratory ({int(new.sum())} newer-drug starts with {int(lab[new > 0].sum())} good responders; observational, confounded)")
K = 8
Zk = Z[:, :K]
X_sh = np.column_stack([C, Zk, new])                           # shared: one RNAseq effect for both drug groups
X_int = np.column_stack([C, Zk, new, Zk * new[:, None]])       # drug x RNAseq terms
Xs = {"clinical + drug": np.column_stack([C, new]), "+ RNAseq (shared effect)": X_sh, "+ RNAseq x drug": X_int}
mae = {}
for name, X in Xs.items():
    P = oof(X, y)
    mae[name] = np.abs(P.mean(0) - y).mean()
    a = np.mean([roc_auc_score(lab, -p) for p in P])
    print(f"   {name:26s} out-of-fold AUC {a:.3f}, MAE of the ratio {np.mean([np.abs(p - y).mean() for p in P]):.3f}")
# predicted ratio under each drug group for each start (the interaction model, refit inside the folds)
pa, pn = np.zeros(len(y)), np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=0).split(X_int):
    m = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(X_int[tr], y[tr])
    for flag, out in ((0.0, pa), (1.0, pn)):
        Xc = X_int[te].copy()
        Xc[:, len(clin) + K] = flag
        Xc[:, len(clin) + K + 1:] = Zk[te] * flag
        out[te] = m.predict(Xc)
diff = pn - pa  # < 0: the newer drugs look better
pref_new = diff < 0
print(f"   predicted ratio, newer minus Adalimumab: mean {diff.mean():+.3f}, sd {diff.std():.3f}; newer drugs look better for {pref_new.mean():.2f} of the starts")
got_new = new > 0
agree = pref_new == got_new
print(f"   the given drug is the predicted best for {agree.mean():.2f} of the starts; realized good share {lab[agree].mean():.3f} (n={agree.sum()}) against {lab[~agree].mean():.3f} (n={(~agree).sum()})")
for nm, s in (("given Adalimumab", ~got_new), ("given newer drug", got_new)):
    print(f"      {nm:18s} predicted best = given: good share {lab[s & agree].mean():.2f} (n={(s & agree).sum()}), otherwise {lab[s & ~agree].mean():.2f} (n={(s & ~agree).sum()})")
n_boot = 300
ci = []
for _ in range(n_boot):
    ix = rng.integers(0, len(y), len(y))
    ci.append(np.nanmean(lab[ix][agree[ix]]) - np.nanmean(lab[ix][~agree[ix]]))
print(f"   good share, given = predicted best minus otherwise: {lab[agree].mean() - lab[~agree].mean():+.3f} [{np.percentile(ci, 2.5):+.3f}, {np.percentile(ci, 97.5):+.3f}] (bootstrap over starts)")

# ---- D. is the PC1 signal real? what is PC1? (the component is chosen inside each training fold, so the test is not optimistic)
from sklearn.linear_model import LogisticRegression
print("\nD. PC1 of the RNAseq embedding")
D["pc1"], D["good"] = Z[:, 0], lab
D["sample_month"] = pd.to_datetime(D.patient_id.map(b.drop_duplicates('patient_id').set_index('patient_id').start_date)).astype("int64") / 8.64e13 / 30.44
all_pc1 = Z_all[:, 0]
print(f"   PC1 of all 600 samples: sd {all_pc1.std():.2f}, skew {pd.Series(all_pc1).skew():+.2f} (a two-humped PC1 would show as skew near 0 and low kurtosis: kurtosis {pd.Series(all_pc1).kurt():+.2f})")
for c in ["B", "mean_all_pre", "hurley", "line", "age", "bmi", "delay", "sample_month"]:
    print(f"   corr(PC1, {c}) = {D['pc1'].corr(D[c].astype(float)):+.2f}")
print("   PC1 by drug: " + ", ".join(f"{k} {D.pc1[drug == k].mean():+.2f}" for k in np.unique(drug)))
print("   PC1 by line: " + ", ".join(f"line {k} {D.pc1[D.line == k].mean():+.2f} (n={int((D.line == k).sum())})" for k in sorted(D.line.unique())))


def cv_top(Xrna, Xc, yb, reps=5, k=1, extra=None):
    """logistic on the k components with the largest |SMD| in the training fold (+ the clinical set if extra)"""
    P = np.zeros((reps, len(yb)))
    for rep in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(Xrna):
            j = np.argsort(-np.abs(smd(Xrna[tr], yb[tr])))[:k]
            A = Xrna[:, j] if extra is None else np.column_stack([extra, Xrna[:, j]])
            P[rep, te] = LogisticRegression(C=1.0, max_iter=500).fit(A[tr], yb[tr]).predict_proba(A[te])[:, 1]
    return P


for name, s in (("all starts", np.ones(len(D), bool)), ("Adalimumab", ada)):
    for lbl, k, extra in (("top 1 component", 1, None), ("top 3 components", 3, None), ("clinical + top 1", 1, C[s])):
        P = cv_top(Z[s], C[s], lab[s], k=k, extra=extra)
        a = np.mean([roc_auc_score(lab[s], p) for p in P])
        print(f"   {name:11s} {lbl:18s} out-of-fold AUC {a:.3f}")
cl = LogisticRegression(C=1.0, max_iter=500)
for name, s in (("all starts", np.ones(len(D), bool)), ("Adalimumab", ada)):
    P = np.zeros((5, s.sum()))
    for rep in range(5):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(C[s]):
            P[rep, te] = cl.fit(C[s][tr], lab[s][tr]).predict_proba(C[s][te])[:, 1]
    print(f"   {name:11s} clinical only (logistic)  out-of-fold AUC {np.mean([roc_auc_score(lab[s], p) for p in P]):.3f}")
# permutation null of the pipeline "top 1 component, Adalimumab"
yb = lab[ada]
real = np.mean([roc_auc_score(yb, p) for p in cv_top(Z[ada], C[ada], yb, reps=3)])
nl = [np.mean([roc_auc_score(ys, p) for p in cv_top(Z[ada], C[ada], ys, reps=1)]) for ys in (rng.permutation(yb) for _ in range(200))]
print(f"   Adalimumab, top 1 component: AUC {real:.3f}; null (200 label shuffles, selection inside): mean {np.mean(nl):.3f}, 95% {np.percentile(nl, 95):.3f}, p = {np.mean(np.array(nl) >= real):.3f}")
