"""
XGBoost version of the treatment-start comparison (counterfactual_treatment.py / plot_counterfactual.py / eval_edit_heterogeneity.py).

Trains XGBoost as in Data_Analysis/xgb_visit_forecast.py (300 trees, learning rate 0.08, depth 6, subsample 0.8, colsample 0.8, hist, missing = -1) on the
training patients of hs_public_dataset/xgb_train.parquet WITHOUT the leaking column time_to_next_days (the table has the same 89 features as the
sequences of my models). Scores it on the test patients (to check it reproduces the table: all rows RMSE 2.957, official rows 2.821), then runs the
same edit test as for my models, on the same biologic starts of the test patients:
  unseen : the prediction at visit k from its own features
  shown  : the same with the seven treatment columns of visit k replaced by those of visit k + 1 (the start is shown)
Writes a cache (runs/xgb/edit_cache.npz) for plot_counterfactual.py: the arrays of the 1,902 starts, and the patient panel of HS028540.
Run with the default python (it has xgboost; torch is not needed).

    python -I -m experiments.latent.xgb_edit [--fast]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PARQUET = "test_data/hs_public_dataset/xgb_train.parquet"
LIGHT = "test_data/hs_public_dataset/hs_sequences_all_light.npz"
TREATMENT_COLS = ["n_tx_started", "days_since_tx_start", "n_tx_ended", "n_tx_active", "Biologic_ever", "Biologic_active", "max_bio_line"]
DROP = ["patient_id", "y", "is_real", "is_test_patient", "time_to_next_days"]


def metrics(p, t):
    e = p - t
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))), "R2": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2))}


def corr_ci(a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.corrcoef(a, b)[0, 1]
    boots = [np.corrcoef(a[j], b[j])[0, 1] for j in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return r, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/xgb")
    ap.add_argument("--patient", default="HS028540")
    ap.add_argument("--cut_month", type=float, default=15.2)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    feats = [c for c in df.columns if c not in DROP]
    tr, te = df.is_test_patient == 0, df.is_test_patient == 1
    model = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=args.n_jobs, missing=-1, random_state=0)
    model.fit(df.loc[tr, feats], df.loc[tr, "y"])
    model.save_model(f"{args.out}/model.json")
    print(f"trained on {tr.sum()} rows, {len(feats)} features in {time.time() - t0:.0f}s", flush=True)
    pred = model.predict(df.loc[te, feats])
    yt, real = df.loc[te, "y"].to_numpy(), df.loc[te, "is_real"].to_numpy() == 1
    for name, sel in (("all rows", np.ones_like(real)), ("official rows (targets.csv)", real), ("derived rows", ~real)):
        m = metrics(pred[sel], yt[sel])
        print(f"XGBoost, {name:28s} ({sel.sum()} rows): RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}", flush=True)

    # ---- the same biologic starts as for my models: from the sequences (raw features, missing = -1 where the mask is 0)
    d = np.load(LIGHT)
    names, off, pid, ym = list(d["feature_names"]), d["offsets"], d["patient_id"], d["y_mask"] > 0
    assert [n for n in names if n not in feats] == [] and [f for f in feats if f not in names] == [], "the feature sets differ"
    X, M = d["X"], d["M"] > 0
    Xr = np.where(M, X, -1.0).astype(np.float32)  # raw values; the table of the parquet has -1 for a missing value
    order = [names.index(f) for f in feats]
    cols = [feats.index(c) for c in TREATMENT_COLS]
    ever = names.index("Biologic_ever")
    pos = {p: i for i, p in enumerate(pid)}
    test_idx = [i for i in range(len(pid)) if d["is_test_patient"][i]]
    events = []
    for i in test_idx:
        a, b = off[i], off[i + 1]
        e = np.where(M[a:b, ever], X[a:b, ever], np.nan)
        events += [(int(i), int(k)) for k in np.flatnonzero(np.diff(e) > 1e-6) if ym[a + k]]
    print(f"{len(events)} biologic starts", flush=True)
    # alignment check of the sequences with the table on the labelled rows of the test patients
    chk = df[te].reset_index(drop=True)
    rows_npz = np.concatenate([np.flatnonzero(ym[off[i]:off[i + 1]]) + off[i] for i in test_idx])
    diff = np.abs(Xr[rows_npz][:, order] - chk[feats].to_numpy(np.float32)).max()
    print(f"alignment: {len(rows_npz)} labelled test rows in the sequences, {len(chk)} in the table, max |difference| of the features {diff:.2e}", flush=True)

    unseen_rows = np.array([off[i] + k for i, k in events])
    A = Xr[unseen_rows][:, order]
    B = A.copy()
    B[:, cols] = Xr[unseen_rows + 1][:, order][:, cols]
    un, se = model.predict(A), model.predict(B)
    y = d["y"][unseen_rows]
    shift, realized = se - un, y - un
    r, lo, hi = corr_ci(shift, realized)
    r0, lo0, hi0 = corr_ci(un, realized)
    res_s = shift - np.polyval(np.polyfit(un, shift, 1), un)
    res_r = realized - np.polyval(np.polyfit(un, realized, 1), un)
    rp, lop, hip = corr_ci(res_s, res_r)
    print(f"\nXGBoost, {len(events)} biologic starts: RMSE vs y(k): unseen {np.sqrt(np.mean((un - y) ** 2)):.3f}   shown {np.sqrt(np.mean((se - y) ** 2)):.3f}")
    print(f"  shift (shown - unseen): mean {shift.mean():+.2f}  median {np.median(shift):+.2f}  share negative {np.mean(shift < 0):.2f}  "
          f"[10%, 90%] = [{np.percentile(shift, 10):+.2f}, {np.percentile(shift, 90):+.2f}]")
    print(f"  true y(k) mean {y.mean():.2f};  mean prediction unseen {un.mean():.2f}, shown {se.mean():.2f}")
    print(f"  corr(predicted shift, realized difference) = {r:+.3f} [{lo:+.3f}, {hi:+.3f}]  slope {np.polyfit(shift, realized, 1)[0]:+.2f}")
    print(f"  control corr(unseen prediction, realized difference) = {r0:+.3f} [{lo0:+.3f}, {hi0:+.3f}]")
    print(f"  with the level taken out of both: {rp:+.3f} [{lop:+.3f}, {hip:+.3f}]")
    edges = np.quantile(shift, np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, shift, side="right") - 1, 0, 9)
    print("  deciles of the shift (mean shift -> mean realized difference):",
          ", ".join(f"{shift[bi == j].mean():+.2f}->{realized[bi == j].mean():+.2f}" for j in range(10)))

    # ---- the patient panel: predictions at every visit, and at the cut and the next visit (shown / hidden)
    i = pos[args.patient]
    a, b = off[i], off[i + 1]
    t = d["time_months"][a:b]
    k_cut = int(np.argmin(np.abs(t - args.cut_month)))
    k_after = next(k for k in range(k_cut + 1, b - a) if X[a + k, ever] > X[a + k_cut, ever] + 1e-6)
    P = Xr[a:b][:, order]
    factual = model.predict(P)
    Pc = P[[k_cut]].copy(); Pc[:, cols] = P[k_after, cols]
    Pa = P[[k_after]].copy(); Pa[:, cols] = P[k_cut, cols]
    patient = {"factual": factual, "un": factual[k_cut], "se": float(model.predict(Pc)[0]), "nat": factual[k_after], "hid": float(model.predict(Pa)[0]),
               "k_cut": k_cut, "k_after": k_after, "y_cut": float(d["y"][a + k_cut])}
    print(f"\n{args.patient}, cut at month {t[k_cut]:.1f} (true {patient['y_cut']:.1f}): start unseen {patient['un']:.1f}, shown {patient['se']:.1f}; "
          f"next visit: real {patient['nat']:.1f}, hidden {patient['hid']:.1f}")
    np.savez(f"{args.out}/edit_cache.npz", un=un, se=se, y=y, factual=factual, k_cut=k_cut, k_after=k_after,
             pat=np.array([patient["un"], patient["se"], patient["nat"], patient["hid"], patient["y_cut"]]))
    print(f"saved {args.out}/edit_cache.npz and {args.out}/model.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
