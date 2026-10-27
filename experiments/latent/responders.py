"""
Exercise 2 on the held-out patients (the targets are known): the GRU against XGBoost, on the patients with an open biologic line.

(a) MAE of both models on the official rows (last visits) where a biologic is active.
(b) Super-responder proxy. Counterfactual outcomes are not observed, so the label here is the proxy
      realized ratio = y / B,   B = mean IHS4 score of the visits in the 6 months before the start of the current biologic line,
      super-responder proxy: y < 0.25 * B  (more than 75% below the pre-treatment level).
    B is the observed pre-start level, not the level the patient would have had untreated: patients start a biologic after a flare, so
    regression to the mean alone lowers y / B. The score of a model is its predicted ratio (prediction / B), low = super-responder.
    Reported: AUC and average precision of -(prediction / B) for the proxy, the rank correlation of the two models' ratios, the overlap
    of the 10% most extreme patients, and a bootstrap over patients for the MAE difference.
The XGBoost model is the flag + 1/sqrt(n) version (the best one), fitted here on the CPU. The GRU predictions are in row_preds.npz.

    python -I experiments/latent/responders.py runs/hs_seq_<stamp of the flag GRU>
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PARQUET = "test_data/hs_public_dataset/xgb_train.parquet"
LIGHT = "test_data/hs_public_dataset/hs_sequences_all_light.npz"
DROP = ["patient_id", "y", "is_real", "is_test_patient", "time_to_next_days"]


def auc(score, label):
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def avg_precision(score, label):
    o = np.argsort(-score)
    hit = label[o]
    return float(np.sum(np.cumsum(hit) / np.arange(1, len(hit) + 1) * hit) / hit.sum())


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def main():
    run = sys.argv[1]
    rp = np.load(f"{run}/row_preds.npz")
    d = np.load(LIGHT)
    names, off, X, M = list(d["feature_names"]), d["offsets"], d["X"], d["M"] > 0
    months = d["time_months"]
    pat_of_row = np.repeat(np.arange(len(d["patient_id"])), np.diff(off))
    rows = rp["rows"]
    real = d["y_is_real"][rows] > 0

    df = pd.read_parquet(PARQUET)
    feats = [c for c in df.columns if c not in DROP]
    tr, te = df[df.is_test_patient == 0], df[df.is_test_patient == 1]
    n = tr.groupby("patient_id").is_real.transform("size")
    w = np.where(tr.is_real == 1, 1.0, 1.0 / np.sqrt((n - 1).clip(lower=1)))
    model = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=4, missing=-1, random_state=0, objective="reg:absoluteerror")
    model.fit(tr[feats + ["is_real"]], tr.y, sample_weight=w / w.mean())
    T = te[feats].copy()
    T["is_real"] = te.is_real.to_numpy()
    px_all = model.predict(T)

    col = lambda nme: names.index(nme)
    sel = np.flatnonzero(real)
    r_glob = rows[sel]
    y, pg, px = d["y"][r_glob], rp["pred"][sel], px_all[sel]
    pid = pat_of_row[r_glob]
    active = np.where(M[r_glob, col("Biologic_active")], X[r_glob, col("Biologic_active")], 0) > 0
    print(f"official rows {len(sel)}; with an active biologic at the last visit: {active.sum()}")

    # (a) MAE on the patients with an open biologic line
    def mae(p, s):
        return float(np.mean(np.abs(p[s] - y[s])))
    rng = np.random.default_rng(0)
    for name, s in (("open biologic line", active), ("no active biologic", ~active)):
        diffs = []
        idx = np.flatnonzero(s)
        eg, ex = np.abs(pg[idx] - y[idx]), np.abs(px[idx] - y[idx])
        for _ in range(500):
            j = rng.integers(0, len(idx), len(idx))
            diffs.append(eg[j].mean() - ex[j].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"(a) {name:20s} n={s.sum():5d}  MAE GRU {mae(pg, s):.3f}  XGBoost {mae(px, s):.3f}  GRU - XGB {mae(pg, s) - mae(px, s):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # (b) the pre-start level B of the current biologic line
    ever, own, own_m = X[:, col("Biologic_ever")], X[:, col("own_score")], M[:, col("own_score")]
    B = np.full(len(sel), np.nan)
    for q, (i, r) in enumerate(zip(pid, r_glob)):
        if not active[q]:
            continue
        a, b = off[i], off[i + 1]
        e = np.where(M[a:b, col("Biologic_ever")], ever[a:b], np.nan)
        ev = np.flatnonzero(np.diff(e) > 1e-6)
        if len(ev) == 0:
            continue
        k = ev[-1]  # row k is the last visit before the start of the current line
        t0 = months[a + k + 1]
        ok = own_m[a:a + k + 1] & (months[a:a + k + 1] >= t0 - 6.0)
        if ok.any():
            B[q] = own[a:a + k + 1][ok].mean()
    have = active & ~np.isnan(B) & (B >= 3)
    print(f"\n(b) open biologic line with a pre-start level B >= 3: {have.sum()} patients (of {active.sum()}; the others have no pre-start scores, or started with the first visit)")
    rr = y[have] / B[have]
    lab = rr < 0.25
    print(f"    realized y / B: median {np.median(rr):.2f}, share with y < 0.25 B (the proxy super-responders): {lab.mean():.3f} ({lab.sum()} patients)")
    rg, rx = pg[have] / B[have], px[have] / B[have]
    for name, r_ in (("GRU", rg), ("XGBoost", rx)):
        print(f"    {name:8s} AUC {auc(-r_, lab):.3f}  AP {avg_precision(-r_, lab):.3f} (base rate {lab.mean():.3f})  "
              f"corr(predicted ratio, realized ratio) {np.corrcoef(r_, rr)[0, 1]:.3f}  predicted ratio < 0.25: {(r_ < 0.25).sum()}")
    k10 = max(1, int(0.1 * have.sum()))
    top_g, top_x = set(np.argsort(rg)[:k10]), set(np.argsort(rx)[:k10])
    print(f"    agreement of the two models: Spearman of the predicted ratios {spearman(rg, rx):.3f}; "
          f"overlap of the {k10} lowest-ratio patients of each: {len(top_g & top_x) / k10:.2f}")
    # label by the baseline, the point-estimate prediction as a score with the level removed: the plain prediction
    print(f"    for comparison, AUC of the plain prediction (no B): GRU {auc(-pg[have], lab):.3f}  XGBoost {auc(-px[have], lab):.3f}")


if __name__ == "__main__":
    main()
