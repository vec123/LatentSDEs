"""
Shifted validation without training: score the trained models on subgroups of the test patients that sit in the tails of the training
distribution of one covariate (lowest / highest 20% of the training rows, or one category). A test 2 that is a shifted population looks
like such a subgroup: the models see feature values that are rare in training. No model is changed or retrained here.
Models: XGBoost (L1, trained here on the CPU as in xgb_edit.py), the GRU of a run with row_preds.npz (save_row_preds.py), and the
persistence baseline (last observed score). Metric: MAE. The interval is a bootstrap over patients (200 draws) of MAE(GRU) - MAE(XGBoost).

    python -I -m experiments.latent.shifted_validation runs/hs_seq_<stamp>      (default python: it has xgboost)
"""
import argparse
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PARQUET = "test_data/hs_public_dataset/xgb_train.parquet"
LIGHT = "test_data/hs_public_dataset/hs_sequences_all_light.npz"
DROP = ["patient_id", "y", "is_real", "is_test_patient", "time_to_next_days"]
COVARIATES = ["age_at_visit", "days_since_diag", "hurley_stage_baseline", "bmi_at_baseline", "n_visits_so_far", "days_since_prev_visit",
              "last_score", "mean_12m", "Biologic_ever", "n_surg", "diagnostic_delay_years", "sex", "smoking_status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_boot", type=int, default=200)
    args = ap.parse_args()
    rp = np.load(f"{args.run}/row_preds.npz")
    d = np.load(LIGHT)
    names, off, ym = list(d["feature_names"]), d["offsets"], d["y_mask"] > 0
    X, M = d["X"], d["M"] > 0
    Xv = np.where(M, X, np.nan)
    istest = d["is_test_patient"].astype(bool)
    n_pat = len(d["patient_id"])
    pat_of_row = np.repeat(np.arange(n_pat), np.diff(off))

    df = pd.read_parquet(PARQUET)
    feats = [c for c in df.columns if c not in DROP]
    tr, te = df.is_test_patient == 0, df.is_test_patient == 1
    model = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=4, missing=-1, random_state=0, objective="reg:absoluteerror")
    model.fit(df.loc[tr, feats], df.loc[tr, "y"])
    px = model.predict(df.loc[te, feats])
    rows = rp["rows"]
    assert len(rows) == te.sum(), (len(rows), te.sum())
    Xr = np.where(M, X, -1.0).astype(np.float32)
    names_f = [names.index(f) for f in feats]
    diff = np.abs(Xr[rows][:, names_f] - df.loc[te, feats].to_numpy(np.float32)).max()
    assert diff < 1e-3, f"the rows of the sequences and the table do not match ({diff})"
    y, real = d["y"][rows], d["y_is_real"][rows] > 0
    pg = rp["pred"]
    last = np.where(M[rows, names.index("last_score")], X[rows, names.index("last_score")], np.nanmean(y))
    pid = pat_of_row[rows]
    tr_rows = np.flatnonzero(ym & ~istest[pat_of_row])  # labelled rows of the training patients: the reference distribution

    def mae(p, sel):
        return float(np.mean(np.abs(p[sel] - y[sel])))

    def boot(sel, rng):
        """cluster bootstrap over patients of MAE(GRU) - MAE(XGBoost) on the selected rows"""
        u, inv = np.unique(pid[sel], return_inverse=True)
        eg = np.bincount(inv, np.abs(pg[sel] - y[sel])); ex = np.bincount(inv, np.abs(px[sel] - y[sel])); cnt = np.bincount(inv)
        out = []
        for _ in range(args.n_boot):
            j = rng.integers(0, len(u), len(u))
            out.append((eg[j].sum() - ex[j].sum()) / cnt[j].sum())
        return np.percentile(out, [2.5, 97.5])

    rng = np.random.default_rng(0)
    groups = [("everything", np.ones(len(y), bool))]
    for c in COVARIATES:
        j = names.index(c)
        ref = Xv[tr_rows, j]; ref = ref[~np.isnan(ref)]
        v = Xv[rows, j]
        lo, hi = np.quantile(ref, [0.2, 0.8])
        if lo == hi:
            continue
        groups += [(f"{c} <= {lo:.3g}", v <= lo), (f"{c} >= {hi:.3g}", v >= hi)]
        miss = np.isnan(v)
        if 0.02 < miss.mean() < 0.98:
            groups.append((f"{c} missing", miss))
    for subset, sel0 in (("OFFICIAL ROWS (4,120 last visits)", real), ("ALL ROWS", np.ones(len(y), bool))):
        print(f"\n===== {subset}: MAE; GRU - XGB = difference with a 95% interval (bootstrap over patients); negative: the GRU is better")
        print(f"{'subgroup':34s} {'rows':>7s} {'XGB L1':>7s} {'GRU L1':>7s} {'persist':>7s}   {'GRU - XGB':>10s}  95% interval")
        for name, g in groups:
            sel = sel0 & g
            if sel.sum() < 100 or len(np.unique(pid[sel])) < 50:
                continue
            a, b = mae(pg, sel), mae(px, sel)
            lo_, hi_ = boot(sel, rng)
            flag = "  *" if hi_ < 0 else ("  !" if lo_ > 0 else "")
            print(f"{name:34s} {sel.sum():7d} {b:7.3f} {a:7.3f} {mae(last, sel):7.3f}   {a - b:+10.3f}  [{lo_:+.3f}, {hi_:+.3f}]{flag}")
    print("\n* GRU better, ! XGBoost better (interval excludes 0). Many groups are compared: expect a few at random.")


if __name__ == "__main__":
    sys.exit(main())
