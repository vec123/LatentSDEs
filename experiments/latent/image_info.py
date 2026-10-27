"""
Does the image embedding hold information about the disease? (CPU, about 1 minute, cross-validation by patient)
Each image belongs to one visit (visit_id), so the IHS4 score at the capture is known. Tests, with 20 principal components of the
256-d embedding (fitted on the 2,800 train images, no label):
 1. the score at the capture visit: history inputs alone / image alone / both (the history uses only visits before the capture);
 2. the mean score of the 6 months after the capture (a forecast target), same models;
 3. the batch effect: can the embedding tell the device? (if yes, the image inputs carry device noise);
 4. the response to a biologic: images taken up to 60 days before a start, against the response ratio of rna_response.py.
Metric: R^2 and MAE out of fold (5 folds by patient, 3 repeats), difference with a 95% bootstrap interval over patients.

    python -I experiments/latent/image_info.py
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

TRAIN = "test_data/hs_public_dataset/public/train/"
rng = np.random.default_rng(0)
v = pd.read_csv(TRAIN + "visits.csv", parse_dates=["visit_date"]).dropna(subset=["ihs4_score"])
demo = pd.read_csv(TRAIN + "demographics.csv").set_index("patient_id")
im = pd.read_csv(TRAIN + "images_metadata.csv", parse_dates=["capture_date"])
emb = pd.read_csv(TRAIN + "images_embeddings.csv")
ecols = [c for c in emb.columns if c != "image_id"]
im = im.merge(emb, on="image_id").merge(v[["visit_id", "ihs4_score"]], on="visit_id", how="left")
Z = PCA(20).fit_transform(im[ecols].to_numpy(float))
Z = Z / Z.std(0)
sc = {p: g for p, g in v.groupby("patient_id")}
rows = []
for k, r in enumerate(im.itertuples()):
    g = sc.get(r.patient_id)
    if g is None:
        continue
    d = (g.visit_date - r.capture_date).dt.days
    prior = g.ihs4_score[(d < 0)]
    p6 = g.ihs4_score[(d < 0) & (d >= -180)]
    nxt = g.ihs4_score[(d > 0) & (d <= 183)]
    dm = demo.loc[r.patient_id]
    rows.append(dict(k=k, patient_id=r.patient_id, now=r.ihs4_score, nxt=nxt.mean() if len(nxt) else np.nan, last=prior.iloc[-1] if len(prior) else np.nan,
                     m6=p6.mean() if len(p6) else np.nan, mall=prior.mean() if len(prior) else np.nan, nprior=len(prior), hurley=dm.hurley_stage_baseline,
                     age=r.age_at_capture, sex=float(r.sex == "Female"), bmi=dm.bmi_at_baseline, device=r.device_id))
D = pd.DataFrame(rows)
zi = D.k.to_numpy()
ZD = Z[zi]
clin = ["last", "m6", "mall", "nprior", "hurley", "age", "sex", "bmi"]
C = D[clin].astype(float)
C = ((C.fillna(C.median()) - C.mean()) / C.std()).to_numpy()
print(f"{len(im)} images, {len(D)} with visits; the score at the capture visit is known for {D.now.notna().sum()}; device ids {D.device.nunique()}")


def oof(X, y, groups, reps=3):
    P = np.zeros((reps, len(y)))
    for rep in range(reps):
        perm = rng.permutation(len(np.unique(groups)))
        gmap = dict(zip(np.unique(groups), perm))
        gg = np.array([gmap[g] for g in groups])
        for tr, te in GroupKFold(5).split(X, y, gg):
            P[rep, te] = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(X[tr], y[tr]).predict(X[te])
    return P.mean(0)


def boot_diff(e_a, e_b, groups, n=1000):
    u, inv = np.unique(groups, return_inverse=True)
    idx = [np.flatnonzero(inv == j) for j in range(len(u))]
    d = e_a - e_b
    m = [d[np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])].mean() for _ in range(n)]
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


for title, col in (("1. score at the capture visit", "now"), ("2. mean score of the 6 months after the capture", "nxt")):
    ok = D[col].notna().to_numpy()
    y, g = D[col].to_numpy()[ok], D.patient_id.to_numpy()[ok]
    res = {}
    for name, X in (("history inputs", C[ok]), ("image alone", ZD[ok]), ("history + image", np.column_stack([C[ok], ZD[ok]]))):
        p = oof(X, y, g)
        res[name] = p
        print(f"{title}: n={ok.sum():5d} {name:16s} R2 {r2_score(y, p):6.3f}  MAE {np.abs(y - p).mean():.3f}")
    d, lo, hi = boot_diff(np.abs(y - res["history inputs"]), np.abs(y - res["history + image"]), g)
    print(f"   MAE gain of the image over the history inputs: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n3. device batch effect (embedding -> device id, logistic, out of fold)")
dev = D.device.to_numpy()
P = np.zeros((len(dev), len(np.unique(dev))))
for tr, te in GroupKFold(5).split(ZD, dev, D.patient_id):
    P[te] = LogisticRegression(max_iter=300).fit(ZD[tr], dev[tr]).predict_proba(ZD[te])
print(f"   accuracy {np.mean(np.argmax(P, 1) == dev):.3f} against chance {np.max(np.bincount(dev)) / len(dev):.3f} (10 devices)")
ok = D.now.notna().to_numpy()
oh = np.eye(10)[dev]
y, g = D.now.to_numpy()[ok], D.patient_id.to_numpy()[ok]
for name, X in (("history + device", np.column_stack([C, oh])), ("history + image + device", np.column_stack([C, ZD, oh]))):
    p = oof(X[ok], y, g)
    print(f"   score at capture, {name:24s} R2 {r2_score(y, p):.3f}  MAE {np.abs(y - p).mean():.3f}")

print("\n4. images shortly before a biologic start and the response")
t = pd.read_csv(TRAIN + "treatments.csv", parse_dates=["start_date"])
t = t[t.treatment_class == "Biologic"]
out = []
for r in D.itertuples():
    row = im.loc[r.k]
    g = sc.get(r.patient_id)
    s = t[(t.patient_id == r.patient_id) & (t.start_date >= row.capture_date) & (t.start_date <= row.capture_date + pd.Timedelta(days=60))]
    if g is None or not len(s):
        continue
    s0 = s.start_date.iloc[0]
    d = (g.visit_date - s0).dt.days
    pre, post = g.ihs4_score[(d >= -180) & (d < 0)], g.ihs4_score[(d >= 90) & (d <= 270)]
    if len(pre) and len(post) and pre.mean() >= 3:
        out.append((r.k, post.mean() / pre.mean(), r.patient_id))
print(f"   {len(out)} starts with an image up to 60 days before and both windows")
if len(out) >= 40:
    kk, ratio, pid = np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])
    good = ratio < 0.5
    Zs = Z[kk]
    Dm = D.set_index("k").loc[kk]
    Cs = Dm[clin].astype(float)
    Cs = ((Cs.fillna(Cs.median()) - Cs.mean()) / Cs.std()).to_numpy()
    print(f"   good responders {good.sum()}")
    for name, X in (("history inputs", Cs), ("image alone", Zs), ("history + image", np.column_stack([Cs, Zs]))):
        p = oof(X, ratio, pid, reps=5)
        print(f"   {name:16s} out-of-fold AUC (good responder) {roc_auc_score(good, -p):.3f}")
