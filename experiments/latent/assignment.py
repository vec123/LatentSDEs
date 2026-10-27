"""
What decides the drug of a biologic start? (default env, CPU, < 2 min.) Unit: each biologic start with a decision step in the train folder.
Target: Adalimumab against a newer drug (Bimekizumab / Secukinumab), and the 3-drug label. Inputs known the day before the start:
the 65 step features (EHR), the calendar day, the image PCs (the latest image before the start; present for some), the RNAseq PCs (the
start-day sample; 600 starts). XGBoost, 5-fold cross-validation by patient, AUC. A feature set that adds AUC over the EHR shows that the
drug choice depends on it, so it is a confounder for any drug comparison.

    python -I experiments/latent/assignment.py
"""
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
names = [str(n) for n in d["feature_names"]]
X, M, kind, day, off = d["X"].astype(np.float32), d["M"] > 0, d["kind"], d["day"].astype(np.int64), d["offsets"]
X = np.where(M, X, np.nan)
prow = np.repeat(np.arange(len(off) - 1), np.diff(off))
bp, bs, bd = d["bio_patient"], d["bio_start"].astype(np.int64), d["bio_drug"]
key = {(int(p), int(s)): int(k) for p, s, k in zip(bp, bs, bd)}
rows, drug = [], []
for r in np.flatnonzero(kind == 3):
    k = key.get((int(prow[r]), int(day[r]) + 1))
    if k is not None:
        rows.append(r); drug.append(k)
rows, drug = np.array(rows), np.array(drug)
pat = prow[rows]
print(f"{len(rows)} starts; Adalimumab {np.mean(drug == 0):.2f}, Bimekizumab {np.mean(drug == 1):.2f}, Secukinumab {np.mean(drug == 2):.2f}")
drop = {"is_decision", "is_query", "is_final", "visit_month", "own_score"}
ehr = [j for j, n in enumerate(names) if n not in drop and not n.startswith("bio_active_") and n not in ("days_on_bio", "cur_bio_line")]
E = X[rows][:, ehr]
cal = day[rows].astype(np.float32)[:, None]
img = d["img_emb"][d["img_row"][rows].clip(0)] if False else None
ir, rr = d["img_row"][rows], d["rna_row"][rows]
ip = PCA(10).fit(d["img_emb"]); rp = PCA(10).fit(d["rna_emb"])
I = np.where((ir >= 0)[:, None], ip.transform(d["img_emb"][ir.clip(0)]), np.nan)
Rn = np.where((rr >= 0)[:, None], rp.transform(d["rna_emb"][rr.clip(0)]), np.nan)
print(f"starts with an image before: {np.mean(ir >= 0):.2f}; with an RNAseq sample: {np.mean(rr >= 0):.3f} ({(rr >= 0).sum()})")


def cv_auc(F, y, sel=None, multi=False):
    sel = np.ones(len(y), bool) if sel is None else sel
    F, y, g = F[sel], y[sel], pat[sel]
    P = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(F, y, g):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=8, verbosity=0)
        P[te] = m.fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    return roc_auc_score(y, P), P, y


new = (drug != 0).astype(int)
print("\nAdalimumab against newer drug (AUC; 0.5 = the inputs say nothing)")
for nm, F, sel in (("calendar day alone", cal, None), ("EHR (no calendar)", E, None), ("EHR + calendar", np.hstack([E, cal]), None)):
    print(f"  {nm:34s} {cv_auc(F, new, sel)[0]:.3f}")
s = ir >= 0
print(f"  -- starts with an image before ({s.sum()}):")
for nm, F in (("EHR + calendar", np.hstack([E, cal])), ("EHR + calendar + image PCs", np.hstack([E, cal, I])), ("image PCs alone", I)):
    print(f"  {nm:34s} {cv_auc(F, new, s)[0]:.3f}")
s = rr >= 0
print(f"  -- starts with an RNAseq sample ({s.sum()}; {int(new[s].sum())} newer drug):")
for nm, F in (("EHR + calendar", np.hstack([E, cal])), ("EHR + calendar + RNAseq PCs", np.hstack([E, cal, Rn])), ("RNAseq PCs alone", Rn)):
    print(f"  {nm:34s} {cv_auc(F, new, s)[0]:.3f}")
m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=8, verbosity=0).fit(np.hstack([E, cal]), new)
nm_all = [names[j] for j in ehr] + ["calendar day"]
top = np.argsort(-m.feature_importances_)[:10]
print("\nstrongest drivers of the choice (gain share): " + ", ".join(f"{nm_all[j]} {m.feature_importances_[j]:.2f}" for j in top))
yr = 2000 + day[rows] / 365.25
for lo, hi in ((2015, 2020), (2020, 2022), (2022, 2024), (2024, 2026)):
    s = (yr >= lo) & (yr < hi)
    if s.sum():
        print(f"  starts {lo}-{hi}: n={s.sum():5d}  Adalimumab {np.mean(drug[s] == 0):.2f}  Bimekizumab {np.mean(drug[s] == 1):.2f}  Secukinumab {np.mean(drug[s] == 2):.2f}")

# ---- Adalimumab is always the first biologic and the newer drugs always follow one, so only Bimekizumab against Secukinumab can be compared
nw = drug != 0
print(f"\nBimekizumab against Secukinumab ({nw.sum()} starts; AUC, 0.5 = the inputs say nothing)")
b2 = (drug == 1).astype(int)
for nm, F in (("calendar day alone", cal), ("EHR (no calendar)", E), ("EHR + calendar", np.hstack([E, cal]))):
    print(f"  {nm:34s} {cv_auc(F, b2, nw)[0]:.3f}")
s = nw & (ir >= 0)
print(f"  starts with an image before ({s.sum()}): image PCs alone {cv_auc(I, b2, s)[0]:.3f}")
s = nw & (rr >= 0)
print(f"  starts with an RNAseq sample ({s.sum()}): RNAseq PCs alone {cv_auc(Rn, b2, s)[0]:.3f}")
