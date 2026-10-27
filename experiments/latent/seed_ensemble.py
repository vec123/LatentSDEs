"""
Seed check and seed ensemble of end-to-end runs (hs_e2e.py): MAE of each run and of their mean on the held-out patients, per group, against XGBoost tuned;
and the forecast from the biologic start (heads_preds.npz of heads_rna.py) where present. Default env, seconds.

    python -I experiments/latent/seed_ensemble.py runs/hs_e2e_<stamp_a> runs/hs_e2e_<stamp_b> [...]
"""
import os
import sys

import numpy as np

runs = sys.argv[1:]
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
flag = d["is_test_patient"] > 0
names = [str(n) for n in d["feature_names"]]
fr = np.array([d["offsets"][i] + np.flatnonzero(d["kind"][d["offsets"][i]:d["offsets"][i + 1]] == 2)[0] for i in np.flatnonzero(flag)])
raw = lambda c: np.where(d["M"][fr, names.index(c)] > 0, d["X"][fr, names.index(c)], np.nan)
groups = {"all held-out patients": np.ones(len(fr), bool), "open biologic line (ex 2)": raw("Biologic_active") > 0,
          "with image (ex 3a)": d["img_row"][fr] >= 0, "with image + RNAseq (ex 3b)": (d["img_row"][fr] >= 0) & (d["rna_row"][fr] >= 0),
          "without visits": np.nan_to_num(raw("n_visits_so_far")) == 0}
F = [dict(np.load(f"{r}/final_preds.npz")) for r in runs]
xg = dict(np.load("runs/xgb_e2e_tuned/final_preds.npz"))
y = F[0]["y"]
for f in F:
    assert (f["patient_id"] == F[0]["patient_id"]).all()
rng = np.random.default_rng(0)


def ci(e):
    m = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(1000)]
    return f"{e.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"


ens = {k: np.mean([f[k] for f in F], 0) for k in ("full_mu", "no_img_mu", "no_rna_mu", "clinical_mu")}
cols = [os.path.basename(r) for r in runs] + ["seed mean", "XGBoost tuned"]
print("MAE at the cutoff (real labels)")
print(f"{'group':30s} {'n':>5s} " + " ".join(f"{c:>24s}" for c in cols))
for g, s in groups.items():
    vals = [np.abs(f["full_mu"][s] - y[s]).mean() for f in F] + [np.abs(ens["full_mu"][s] - y[s]).mean(), np.abs(xg["full_mu"][s] - y[s]).mean()]
    print(f"{g:30s} {s.sum():5d} " + " ".join(f"{v:24.3f}" for v in vals))
print("\npaired differences, 95% interval over patients")
for g, s in groups.items():
    e_a, e_b, e_m, e_x = (np.abs(v[s] - y[s]) for v in (F[0]["full_mu"], F[1]["full_mu"], ens["full_mu"], xg["full_mu"]))
    print(f"   {g:30s} run b - run a {ci(e_b - e_a)}   seed mean - run a {ci(e_m - e_a)}   seed mean - XGBoost {ci(e_m - e_x)}")
print(f"\nagreement of the two runs: corr of predictions {np.corrcoef(F[0]['full_mu'], F[1]['full_mu'])[0, 1]:.4f}, "
      f"mean |difference| {np.abs(F[0]['full_mu'] - F[1]['full_mu']).mean():.3f}")
for g in ("with image (ex 3a)", "with image + RNAseq (ex 3b)"):
    s = groups[g]
    print(f"ex 3 gains, {g}: " + "; ".join(f"{nm} clinical {np.abs(v['clinical_mu'][s] - y[s]).mean():.3f} -> image {np.abs(v['no_rna_mu'][s] - y[s]).mean():.3f} "
                                               f"-> all {np.abs(v['full_mu'][s] - y[s]).mean():.3f}" for nm, v in zip(["run a", "run b", "seed mean"], F[:2] + [ens])))
H = [f"{r}/heads_preds.npz" for r in runs]
if all(os.path.exists(h) for h in H):
    H = [dict(np.load(h)) for h in H]
    yh, rna = H[0]["y"], H[0]["has_rna"]
    print("\nforecast from the biologic start (head 2, decision step, given drug), MAE")
    for nm, s in (("all open lines", np.ones(len(yh), bool)), ("with RNAseq", rna)):
        vals = {f"run {c}": np.abs(h["full_h2start"][s] - yh[s]).mean() for c, h in zip("ab", H)}
        m = np.mean([h["full_h2start"] for h in H], 0)
        vals["seed mean"] = np.abs(m[s] - yh[s]).mean()
        gain = np.mean([h["img_h2start"] for h in H], 0)
        vals["RNAseq gain (seed mean)"] = np.abs(gain[s] - yh[s]).mean() - vals["seed mean"]
        print(f"   {nm:15s} n={s.sum():4d}  " + "  ".join(f"{k} {v:.3f}" for k, v in vals.items()))
