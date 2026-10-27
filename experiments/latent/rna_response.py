"""
Does the baseline RNAseq sample predict the response to a biologic? (CPU, cross-validation over patients, about 1 minute)

Unit: the 600 biologic starts that have an RNAseq sample taken on the day of the start (all 600 samples are such baseline samples).
Response: mean IHS4 score in days 90..270 after the start divided by B, the mean score in the 180 days before the start (B >= 3 and
at least one score in both windows). "Good responder": ratio < 0.5. The inputs are known on the start day (the score history before
it, the demographics, the drug and the line), plus, for the last model, the 128-d RNAseq embedding.
Models (5-fold cross-validation by patient, 5 repeats): the pre-start level B alone; the clinical set; clinical + RNAseq.
Metrics: AUC of the good-responder label and R^2 of the ratio, as mean over repeats with the spread over repeats.

    python -I experiments/latent/rna_response.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

TRAIN = "C:/Users/vic-b/Documents/Victors/Projects/Data_Analysis/hs_public_dataset/public/train/"
t = pd.read_csv(TRAIN + "treatments.csv", parse_dates=["start_date"])
v = pd.read_csv(TRAIN + "visits.csv", parse_dates=["visit_date"]).dropna(subset=["ihs4_score"])
demo = pd.read_csv(TRAIN + "demographics.csv").set_index("patient_id")
rm = pd.read_csv(TRAIN + "rnaseq_metadata.csv", parse_dates=["collection_date"])
re_ = pd.read_csv(TRAIN + "rnaseq_embeddings.csv").merge(rm[["rnaseq_sample_id", "patient_id", "collection_date"]], on="rnaseq_sample_id")
emb_cols = [c for c in re_.columns if c.startswith("rnaseq_dim_")]

b = t[t.treatment_class == "Biologic"].merge(re_[["patient_id", "collection_date"] + emb_cols], on="patient_id")
b = b[b.start_date == b.collection_date].drop_duplicates("patient_id")  # the start the sample belongs to
print(f"{len(b)} baseline RNAseq samples matched to a biologic start; drugs: {b.drug_name.value_counts().to_dict()}")
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
    rows.append(dict(patient_id=r.patient_id, ratio=post.mean() / pre.mean(), B=pre.mean(), n_pre=len(pre), sd_pre=pre.std() if len(pre) > 1 else 0.0,
                     last_pre=g.ihs4_score[d < 0].iloc[-1], mean_all_pre=prior.mean(), n_all_pre=len(prior),
                     age=r.start_date.year - dm.birth_year, sex=float(dm.sex == "Female"), bmi=dm.bmi_at_baseline, hurley=dm.hurley_stage_baseline,
                     delay=dm.diagnostic_delay_years, smoke=float(dm.smoking_status == "Current") if isinstance(dm.smoking_status, str) else np.nan,
                     line=r.treatment_line, ada=float(r.drug_name == "Adalimumab"), bime=float(r.drug_name == "Bimekizumab"),
                     **{c: getattr(r, c) for c in emb_cols}))
D = pd.DataFrame(rows)
y = D.ratio.to_numpy()
lab = y < 0.5
print(f"{len(D)} starts with both windows; median ratio {np.median(y):.2f}; good responders (ratio < 0.5): {lab.sum()} ({lab.mean():.2f})")
clin = ["B", "n_pre", "sd_pre", "last_pre", "mean_all_pre", "n_all_pre", "age", "sex", "bmi", "hurley", "delay", "smoke", "line", "ada", "bime"]
sets = {"pre-start level B only": ["B"], "clinical": clin, "clinical + RNAseq (128-d)": clin + emb_cols, "RNAseq (128-d) alone": emb_cols}
Xall = D[clin + emb_cols].astype(float)
Xall = Xall.fillna(Xall.median())
mu, sd = Xall.mean(), Xall.std().replace(0, 1)
Z = ((Xall - mu) / sd).to_numpy()
cols = list(Xall.columns)
print(f"\n{'inputs':32s} {'AUC (good responder)':>22s}  {'R2 of the ratio':>16s}")
for name, cs in sets.items():
    idx = [cols.index(c) for c in cs]
    auc, r2 = [], []
    for rep in range(5):
        p = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(Z):
            m = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Z[tr][:, idx], y[tr])
            p[te] = m.predict(Z[te][:, idx])
        auc.append(roc_auc_score(lab, -p)); r2.append(r2_score(y, p))
    print(f"{name:32s} {np.mean(auc):10.3f} (+-{np.std(auc):.3f})  {np.mean(r2):9.3f} (+-{np.std(r2):.3f})")
# the strongest single RNAseq dimensions against the ratio, and how many would be expected by chance
rs = np.array([np.corrcoef(D[c], y)[0, 1] for c in emb_cols])
print(f"\nRNAseq dimensions vs response ratio: largest |r| {np.abs(rs).max():.3f}; dimensions with |r| > {2 / np.sqrt(len(D)):.3f}: "
      f"{(np.abs(rs) > 2 / np.sqrt(len(D))).sum()} of {len(rs)} (about {0.05 * len(rs):.0f} expected by chance)")

# ---- which treatment, and how wide are the embeddings
D["drug"] = np.where(D.ada == 1, "Adalimumab", np.where(D.bime == 1, "Bimekizumab", "Secukinumab"))
D["good"] = lab
print("\nstarts used, by drug (n, good responders, median ratio):")
print(D.groupby("drug").agg(n=("good", "size"), good=("good", "sum"), median_ratio=("ratio", "median")).round(2).to_string())
print("by treatment line (n, good responders):")
print(D.groupby("line").agg(n=("good", "size"), good=("good", "sum")).to_string())
ab = t[t.treatment_class == "Antibiotic"].copy()
ab["end_date"] = pd.to_datetime(ab.end_date)
act = []
for r in D.merge(b[["patient_id", "start_date"]], on="patient_id").itertuples():
    a = ab[(ab.patient_id == r.patient_id) & (ab.start_date <= r.start_date) & (ab.end_date.isna() | (ab.end_date >= r.start_date))]
    act.append(len(a) > 0)
D["abx"] = act
print(f"an antibiotic is active on the start day: {D.abx.mean():.2f} of the starts; good responders {D[D.abx].good.mean():.2f} with, {D[~D.abx].good.mean():.2f} without")
im = pd.read_csv(TRAIN + "images_embeddings.csv", nrows=2)
print(f"image embedding columns: {len([c for c in im.columns if c != 'image_id'])}; RNAseq embedding columns: {len(emb_cols)}")
