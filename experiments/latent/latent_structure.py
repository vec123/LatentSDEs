"""
Structure of the latents of a patient-code model (hs_fused_sde.py --code_dim; no training; GPU in chunks of 50 patients, < 1 min).

A. The patient code c (the trait; trained for the training patients only): which patient features it encodes (cross-validated R2 of a ridge
   from c to each feature), its principal components, and clusters (Gaussian mixture, k by BIC) with their profiles.
B. Responders: the biologic starts of the training patients (hs_groups.find_starts: ratio = mean score days 90..270 after the start / mean
   180 days before, pre mean >= 3; good responder: ratio < 0.5). Out-of-fold AUC (logistic regression, 5 folds grouped by patient) of the
   start features (hs_task2.PROFILE + drug), the code c, the posterior mean m at the decision step (16 PCs of the 128-d state), and their
   combinations. c is fitted on the whole record, also after the start: descriptive only. m at the decision step reads no later data:
   predictive. Earlier (section 9.1): start features alone 0.729 on held-out starts.

    python -m experiments.latent.latent_structure [runs/hs_fused_sde_<stamp>]
"""
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.latent.hs_e2e import PATHS, VISIT, Data
from experiments.latent.hs_fused_sde import make_model, prepare
from experiments.latent.hs_groups import find_starts
from experiments.latent.hs_task2 import PROFILE

DEMO = "test_data/hs_public_dataset/public/train/demographics.csv"
YEAR = 365.25


def features(data, pats, demo):
    nm = data.names.index
    rows = []
    for i in pats:
        s, r0, rf = slice(data.off[i], data.off[i + 1]), data.off[i], data.off[i + 1] - 1
        years = max((data.day[rf] - data.day[r0]) / YEAR, 0.25)
        sc = data.score[s][data.has_score[s]]
        dm = demo.loc[data.pid[i]]
        val = lambda c, r: data.raw[r, nm(c)] if data.M[r, nm(c)] else np.nan
        rows.append({"hurley stage": val("hurley_stage_baseline", r0), "female": float(dm.sex == "Female"),
                     "black": float(dm.ethnicity == "Black/African American"), "bmi": val("bmi_at_baseline", r0),
                     "current smoker": float(dm.smoking_status == "Current"), "age at diagnosis": val("age_at_diagnosis", r0),
                     "family history": float(dm.family_history_hs == "Yes"),
                     "mean visit score": sc.mean() if len(sc) else np.nan, "sd of visit scores": sc.std() if len(sc) > 2 else np.nan,
                     "flare visits per year": (data.vt[s] == 1).sum() / years,
                     "share on a biologic": (data.raw[s, nm("Biologic_active")] > 0).mean(),
                     "share on an antibiotic": (data.raw[s, nm("Antibiotic_active")] > 0).mean(),
                     "real label (6-month mean)": data.y[rf]})
    return pd.DataFrame(rows)


def oof_auc(X, y, groups, seed=0):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X, y,
                          cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    rng = np.random.default_rng(seed)
    up = np.unique(groups)
    bs = []
    for _ in range(300):
        sel = np.isin(groups, rng.choice(up, len(up)))
        if y[sel].min() != y[sel].max():
            bs.append(roc_auc_score(y[sel], p[sel]))
    return roc_auc_score(y, p), np.percentile(bs, [2.5, 97.5]), p


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_fused_sde_20261026-234554"
    ck = torch.load(f"{run}/model.pt", map_location="cuda", weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"]).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    pool = np.random.default_rng(ck["args"].get("seed", 0)).permutation(np.flatnonzero(~data.flag))
    tr = np.sort(pool[int(0.1 * len(pool)):])
    demo = pd.read_csv(DEMO).set_index("patient_id")
    C = model.pcode.weight.detach().cpu().numpy()

    print(f"model {run}: code size {C.shape[1]}, {len(tr)} training patients")
    F = features(data, tr, demo)
    Ct = C[tr]
    print("\nA. What the patient code encodes: cross-validated R2 (ridge, 5 folds) from c to each feature; Spearman with the first 3 PCs of c")
    Z = StandardScaler().fit_transform(Ct)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = U[:, :3] * S[:3]
    ev = S ** 2 / (S ** 2).sum()
    print(f"   variance of the code PCs: " + " ".join(f"{v:.2f}" for v in ev[:6]) + " ...")
    for col in F.columns:
        f = F[col].to_numpy(float)
        ok = np.isfinite(f)
        pr = cross_val_predict(RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)), Z[ok], f[ok], cv=KFold(5, shuffle=True, random_state=0))
        r2 = 1 - ((f[ok] - pr) ** 2).sum() / ((f[ok] - f[ok].mean()) ** 2).sum()
        print(f"   {col:28s} R2 {r2:+.3f}   rho with PC1 / PC2 / PC3: " + " / ".join(f"{spearmanr(pcs[ok, k], f[ok])[0]:+.2f}" for k in range(3)))
    print("\n   Gaussian mixture on the standardized code (full covariance): BIC by k")
    bic = {k: GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(Z).bic(Z) for k in range(1, 9)}
    print("   " + "  ".join(f"k={k}: {b:.0f}" for k, b in bic.items()))
    kb = min(bic, key=bic.get)
    if kb > 1:
        lab = GaussianMixture(kb, covariance_type="full", random_state=0, n_init=2).fit_predict(Z)
        print(f"   best k = {kb}; cluster profiles (mean of each feature):")
        print("   " + F.groupby(lab).mean().round(2).assign(n=np.bincount(lab)).T.to_string().replace("\n", "\n   "))
    else:
        print("   best k = 1: the code forms one continuous cloud, no separate clusters")

    st = find_starts(data, tr)
    pat = np.array([s[0] for s in st])
    good = np.array([s[4] < 0.5 for s in st])
    drug = np.array([s[2] for s in st])
    rows = np.array([data.off[s[0]] + s[3] for s in st])
    nm = data.names.index
    B = np.column_stack([np.where(data.M[rows, nm(c)], data.raw[rows, nm(c)], np.nan) for c in PROFILE] +
                        [(drug == d).astype(float) for d in np.unique(drug)])
    B = np.where(np.isfinite(B), B, np.nanmedian(B, 0))
    M = np.zeros((len(st), model.dz), np.float32)
    up = np.unique(pat)
    with torch.no_grad():
        for a in range(0, len(up), 50):
            sel = [int(i) for i in up[a:a + 50]]
            model.encode(data.pack(sel, "cuda"))
            m = model.last_post[0].cpu().numpy()
            for b, i in enumerate(sel):
                for j in np.flatnonzero(pat == i):
                    M[j] = m[b, st[j][3]]
    Mz = StandardScaler().fit_transform(M)
    Um, Sm, _ = np.linalg.svd(Mz - Mz.mean(0), full_matrices=False)
    M16 = Um[:, :16] * Sm[:16]
    Cs = C[pat]
    print(f"\nB. Responders: {len(st)} biologic starts of {len(up)} training patients; good responders {good.mean():.3f} (ratio < 0.5)")
    print("   out-of-fold AUC (logistic, 5 folds grouped by patient) [95% bootstrap over patients]")
    res = {}
    for name, X in (("start features", B), ("code c (descriptive: c saw the data after the start)", Cs),
                    ("state m at the decision step (16 PCs; predictive)", M16), ("start features + m", np.column_stack([B, M16])),
                    ("start features + c (descriptive)", np.column_stack([B, Cs])),
                    ("start features + m + c (descriptive)", np.column_stack([B, M16, Cs]))):
        a, ci, p = oof_auc(X, good, pat)
        res[name] = p
        print(f"   {name:56s} AUC {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    d = (Cs[good].mean(0) - Cs[~good].mean(0)) / Cs.std(0)
    print(f"   code: standardized difference good - other, largest |d| {np.abs(d).max():.2f} (dim {np.abs(d).argmax()}), mean |d| {np.abs(d).mean():.2f}")
    print(f"   Spearman of the response ratio with code PC1 / PC2 / PC3 (PCs of all training codes): " +
          " / ".join(f"{spearmanr((((Cs - Ct.mean(0)) / Ct.std(0)) @ Vt[k]), [s[4] for s in st])[0]:+.2f}" for k in range(3)))


if __name__ == "__main__":
    main()
