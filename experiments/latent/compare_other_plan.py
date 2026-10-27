"""
Paired comparison of the other team's exercise 1 model (other_plan/round1 (1).py, "HSBreakers") with the GRU (hs_e2e.py) and XGBoost (xgb_e2e.py) on the
same 4,203 held-out patients of the train folder. Their feature table (full_features + practitioner statistics pooled over train / test_1 / test_2 +
shrunk level; ethnicity and practitioner id dropped, as they do) and their gradient boosting (HistGradientBoosting, absolute error, their parameters) are
trained on our training patients (the 80% that the GRU and XGBoost use) and predict our held-out patients. Real labels at the cutoff, MAE, 95% interval
from a bootstrap over patients.
Deviation from their final model (time): one fit with early stopping, no thinned copies and no seed average (their CV: thinned copies 1.457 -> 1.440).

    python -I experiments/latent/compare_other_plan.py [runs/hs_e2e_<stamp>]
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("test_data/hs_public_dataset/public")
OUT = Path("runs/other_plan")
OUT.mkdir(parents=True, exist_ok=True)
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_e2e_20261025-085104"
spec = importlib.util.spec_from_file_location("hsb", "other_plan/round1 (1).py")
hsb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsb)

t0 = time.time()
cache = OUT / "features_all.pkl"
if cache.exists():
    ALL = pd.read_pickle(cache)
else:
    names = ("train", "test_1", "test_2")
    T = {k: hsb.load_split(ROOT / k) for k in names}
    F = {k: hsb.full_features(T[k]) for k in names}
    ALL = hsb.add_practitioner_stats(pd.concat([F[k].assign(split=k) for k in names]))
    ALL["shrunk_level"] = hsb.shrunk_level(ALL)
    for c in ALL.select_dtypes("object").columns:
        ALL[c] = ALL[c].astype("category")
    ALL.to_pickle(cache)
print(f"features: {time.time() - t0:.0f}s")
X = ALL[ALL.split == "train"].drop(columns="split")
y = pd.read_csv(ROOT / "train" / "targets.csv").set_index("patient_id").avg_ihs4_next6.reindex(X.index)
feats = [c for c in X.columns if c not in ("practitioner", "ethnicity")]
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
pid, flag = d["patient_id"].astype(str), d["is_test_patient"] > 0
tr_ids, te_ids = pid[~flag], pid[flag]
print(f"{len(feats)} features; {len(tr_ids)} training patients, {len(te_ids)} held-out patients")
t1 = time.time()
m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5.0,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, categorical_features="from_dtype", random_state=0)
m.fit(X.loc[tr_ids, feats], y.loc[tr_ids])
p_hsb = np.clip(m.predict(X.loc[te_ids, feats]), 0, None)
print(f"fit: {time.time() - t1:.0f}s, {m.n_iter_} iterations")
np.savez(OUT / "heldout_preds.npz", patient_id=te_ids, pred=p_hsb)

g = dict(np.load(f"{RUN}/final_preds.npz"))
xg = dict(np.load("runs/xgb_e2e_tuned/final_preds.npz"))
assert (g["patient_id"].astype(str) == te_ids).all() and (xg["patient_id"].astype(str) == te_ids).all()
yt = g["y"]
assert np.allclose(yt, y.loc[te_ids].values)
names = [str(n) for n in d["feature_names"]]
fr = np.array([d["offsets"][i] + np.flatnonzero(d["kind"][d["offsets"][i]:d["offsets"][i + 1]] == 2)[0] for i in np.flatnonzero(flag)])
raw = lambda c: np.where(d["M"][fr, names.index(c)] > 0, d["X"][fr, names.index(c)], np.nan)
groups = {"all held-out patients": np.ones(len(te_ids), bool), "open biologic line (ex 2)": raw("Biologic_active") > 0,
          "with image (ex 3a)": d["img_row"][fr] >= 0, "with image + RNAseq (ex 3b)": (d["img_row"][fr] >= 0) & (d["rna_row"][fr] >= 0),
          "without visits": np.nan_to_num(raw("n_visits_so_far")) == 0}
P = {"HSBreakers GBM": p_hsb, "GRU": g["full_mu"], "XGBoost tuned": xg["full_mu"], "GRU + HSBreakers / 2": (g["full_mu"] + p_hsb) / 2,
     "GRU + XGBoost / 2": (g["full_mu"] + xg["full_mu"]) / 2}
rng = np.random.default_rng(0)


def ci(e):
    m_ = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(1000)]
    return f"{e.mean():+.3f} [{np.percentile(m_, 2.5):+.3f}, {np.percentile(m_, 97.5):+.3f}]"


print(f"\nMAE on the held-out patients (real labels)")
print(f"{'group':30s} {'n':>5s} " + " ".join(f"{k:>21s}" for k in P))
for gname, s in groups.items():
    print(f"{gname:30s} {s.sum():5d} " + " ".join(f"{np.abs(v[s] - yt[s]).mean():21.3f}" for v in P.values()))
print("\npaired differences (MAE first - MAE second; < 0: the first is better), 95% interval over patients")
for a, b in (("GRU", "HSBreakers GBM"), ("HSBreakers GBM", "XGBoost tuned"), ("GRU + HSBreakers / 2", "GRU")):
    for gname, s in groups.items():
        print(f"   {a:22s} - {b:16s} {gname:30s} {ci(np.abs(P[a][s] - yt[s]) - np.abs(P[b][s] - yt[s]))}")

# ---------------- exercise 3 as they do it: a PCA + ridge correction of the clinical model's residual, fitted on training patients with the embeddings
# (out-of-fold residuals of their model), applied only if the cross-validated gain has a 95% interval above 0. 3a: only patients on a biologic whose line
# started recently (first third of days on the biologic); 3b: image + RNAseq stacked.
t2 = time.time()
oof_file = OUT / "oof_train.npz"
if oof_file.exists():
    oof = pd.Series(dict(np.load(oof_file))["oof"], tr_ids)
else:
    oof_arr, _ = hsb.cv_lgb(X.loc[tr_ids, feats], y.loc[tr_ids])
    np.savez(oof_file, oof=oof_arr)
    oof = pd.Series(oof_arr, tr_ids)
print(f"\nout-of-fold predictions of their model on our training patients: {time.time() - t2:.0f}s (MAE {np.abs(oof - y.loc[tr_ids]).mean():.3f})")
EMB = hsb.load_embeddings(ROOT / "train")
base_te = pd.Series(p_hsb, te_ids)
ic = [c for c in EMB["img"].columns if c.startswith("image_dim_")]
# 3b
both = EMB["img"].index.intersection(EMB["rna"].index)
itr, ite = both.intersection(pd.Index(tr_ids)), both.intersection(pd.Index(te_ids))
ptr = [EMB["img"].loc[itr, ic].values, EMB["rna"].loc[itr, EMB["rna_dims"]].values]
pte = [EMB["img"].loc[ite, ic].values, EMB["rna"].loc[ite, EMB["rna_dims"]].values]
Etr, Ete = hsb.stack_embeddings(ptr, pte)
gain, gci = hsb.emb_gain(y.loc[itr].values, oof.loc[itr].values, Etr)
corr = hsb.embedding_correction(Etr, y.loc[itr].values - oof.loc[itr].values, Ete)
p3b = base_te.copy()
if gci[0] > 0:
    p3b.loc[ite] = np.clip(base_te.loc[ite].values + corr, 0, None)
print(f"3b: {len(itr)} training patients, CV gain {gain:+.3f} [{gci[0]:+.3f}, {gci[1]:+.3f}] -> applied {gci[0] > 0}")
# 3a (recent biologic starters)
iim = EMB["img"].index
itr_i = iim.intersection(pd.Index(tr_ids))
on, bd = X.loc[itr_i, "on_bio"].values == 1, X.loc[itr_i, "bio_days"].values
thr = np.nanquantile(bd[on], 1 / 3)
rec = itr_i[on & (bd <= thr)]
g3a, c3a = hsb.emb_gain(y.loc[rec].values, oof.loc[rec].values, EMB["img"].loc[rec, ic].values)
ite_i = iim.intersection(pd.Index(te_ids))
r_te = ite_i[(X.loc[ite_i, "on_bio"].values == 1) & (X.loc[ite_i, "bio_days"].values <= thr)]
p3a = base_te.copy()
if c3a[0] > 0 and len(r_te):
    p3a.loc[r_te] = np.clip(base_te.loc[r_te].values + hsb.embedding_correction(EMB["img"].loc[rec, ic].values, y.loc[rec].values - oof.loc[rec].values,
                                                                                  EMB["img"].loc[r_te, ic].values), 0, None)
print(f"3a: {len(rec)} recent starters in training (<= {thr:.0f} days), CV gain {g3a:+.3f} [{c3a[0]:+.3f}, {c3a[1]:+.3f}] -> applied {c3a[0] > 0}; "
      f"{len(r_te)} held-out patients corrected")
P3 = {"HSBreakers ex3 (3a / 3b model)": None, "GRU": g["full_mu"], "XGBoost tuned": xg["full_mu"]}
for gname, s, pp in (("with image (ex 3a)", groups["with image (ex 3a)"], p3a.values), ("with image + RNAseq (ex 3b)", groups["with image + RNAseq (ex 3b)"], p3b.values)):
    e_h, e_g, e_x = np.abs(pp[s] - yt[s]), np.abs(g["full_mu"][s] - yt[s]), np.abs(xg["full_mu"][s] - yt[s])
    print(f"   {gname:30s} n={s.sum():4d}  HSBreakers {e_h.mean():.3f}  GRU {e_g.mean():.3f}  XGBoost {e_x.mean():.3f}   GRU - HSBreakers {ci(e_g - e_h)}")
r_mask = np.isin(te_ids, r_te)
if r_mask.any():
    print(f"   recent biologic starters with image (their 3a target group) n={r_mask.sum()}: HSBreakers {np.abs(p3a.values[r_mask] - yt[r_mask]).mean():.3f}  "
          f"GRU {np.abs(g['full_mu'][r_mask] - yt[r_mask]).mean():.3f}  GRU - HSBreakers {ci(np.abs(g['full_mu'][r_mask] - yt[r_mask]) - np.abs(p3a.values[r_mask] - yt[r_mask]))}")

# ---------------- exercise 2: their super-responder label and the RNAseq AUC (their 0.93 is a 5-fold CV inside the 600 RNAseq patients)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
ratio, lab075, labrate = hsb.super_responder_pseudolabels(X, y)
label = labrate
ids_r = EMB["rna"].index.intersection(label.index)
R = EMB["rna"].loc[ids_r, EMB["rna_dims"]].values
Zr = PCA(16, random_state=0).fit_transform((R - R.mean(0)) / R.std(0))
lab_r = label.loc[ids_r].values
cv = np.mean([roc_auc_score(lab_r, cross_val_predict(LogisticRegression(C=0.1, max_iter=2000), Zr, lab_r, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                                     method="predict_proba")[:, 1]) for s in range(3)])
print(f"\nex 2: their label on {len(ids_r)} RNAseq patients ({lab_r.sum()} positives): RNAseq AUC, their 5-fold CV {cv:.3f}")
trm, tem = np.isin(ids_r, tr_ids), np.isin(ids_r, te_ids)
pipe = LogisticRegression(C=0.1, max_iter=2000).fit(Zr[trm], lab_r[trm])
print(f"   on our split: fit on {trm.sum()} training RNAseq patients, AUC on {tem.sum()} held-out ({lab_r[tem].sum()} positives): "
      f"{roc_auc_score(lab_r[tem], pipe.predict_proba(Zr[tem])[:, 1]):.3f}")
# what defines the label: the real 2026 label of the patient divided by an untreated prediction -> it partly re-encodes the outcome itself
print(f"   corr(label ratio, real label y) on biologic patients: {np.corrcoef(ratio.values, y.loc[ratio.index].values)[0, 1]:+.3f}")
if r_mask.any():
    print(f"   their 3a correction out of sample (119 recent starters): base {np.abs(base_te.values[r_mask] - yt[r_mask]).mean():.3f} -> corrected "
          f"{np.abs(p3a.values[r_mask] - yt[r_mask]).mean():.3f}")
s3b = groups["with image + RNAseq (ex 3b)"]
print(f"   their 3b correction out of sample (114): base {np.abs(base_te.values[s3b] - yt[s3b]).mean():.3f} -> corrected {np.abs(p3b.values[s3b] - yt[s3b]).mean():.3f}")
# ranking of their super-responder label on the held-out biologic patients: their pre-treatment classifier (fitted on our training patients) against the GRU
# forecast at the cutoff divided by the same untreated prediction (the GRU also reads the visits after the start, as the exercise allows)
lab_tr = label[label.index.isin(tr_ids)]
te_bio = label.index[label.index.isin(te_ids)]
prob, auc_cv, _ = hsb.super_responder_model(X, lab_tr, X.loc[te_bio])
cf = (y.loc[ratio.index] / ratio).loc[te_bio]
gru = pd.Series(g["full_mu"], te_ids).loc[te_bio] / cf.clip(lower=0.5)
lt = label.loc[te_bio].values
print(f"\nex 2 ranking of their label on {len(te_bio)} held-out biologic patients ({lt.sum()} positives): their classifier AUC {roc_auc_score(lt, prob.values):.3f} "
      f"(CV on training {auc_cv:.3f}); GRU forecast / untreated prediction AUC {roc_auc_score(lt, -gru.values):.3f}")
print("\nblends on the exercise 3 groups (MAE)")
for gname, s, pp in (("with image (ex 3a)", groups["with image (ex 3a)"], p3a.values), ("with image + RNAseq (ex 3b)", s3b, p3b.values)):
    bl = (g["full_mu"] + pp) / 2
    bl3 = (g["full_mu"] + pp + xg["full_mu"]) / 3
    print(f"   {gname:30s} GRU {np.abs(g['full_mu'][s] - yt[s]).mean():.3f}  (GRU + HSBreakers ex3) / 2 {np.abs(bl[s] - yt[s]).mean():.3f}  "
          f"(GRU + HSBreakers ex3 + XGBoost) / 3 {np.abs(bl3[s] - yt[s]).mean():.3f}   blend/2 - GRU {ci(np.abs(bl[s] - yt[s]) - np.abs(g['full_mu'][s] - yt[s]))}")
np.savez(OUT / "heldout_preds.npz", patient_id=te_ids, pred=p_hsb, pred_ex3a=p3a.values, pred_ex3b=p3b.values)
