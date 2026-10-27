"""
XGBoost baseline for the forecast from the biologic start (the counterpart of head 2 at the decision step, see heads_rna.py): one row
per patient with an open biologic line at the cutoff; the features of the decision step of the current line (the day before its start;
it sees the samples of the start day), the drug and line of that start, and the months from the decision day to the cutoff. Target:
the real label (mean IHS4 of Jan-Jun 2026). Three separately trained models: clinical only, + image, + image + RNAseq (16 PCs each, PCA
fitted on the train folder). Early stopping on the validation patients of hs_e2e.py. Scored on the held-out patients with an open line,
paired with the GRU (<gru>/heads_preds.npz, written by heads_rna.py) and with the XGBoost cutoff forecast (<xgb>/final_preds.npz).

    python -I experiments/latent/xgb_start.py [--gru runs/hs_e2e_<stamp>] [--xgb runs/xgb_e2e_tuned]
"""
import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

TRAIN = "test_data/hs_public_dataset/hs_e2e_train.npz"
FINAL, DECISION = 2, 3
MONTH = 30.44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru", default="runs/hs_e2e_20261025-004126")
    ap.add_argument("--xgb", default="runs/xgb_e2e_tuned")
    ap.add_argument("--out", default="runs/xgb_start")
    ap.add_argument("--n_jobs", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = dict(np.load(TRAIN))  # loaded once (an npz decompresses an array at every access)
    names = [str(n) for n in d["feature_names"]]
    keep = [j for j, n in enumerate(names) if n != "is_decision"]
    off, kind, day = d["offsets"], d["kind"], d["day"]
    cut = int(day[kind == FINAL].max())
    bp, bs, be, bd, bl = d["bio_patient"], d["bio_start"], d["bio_end"], d["bio_drug"], d["bio_line"]
    order = np.argsort(bp, kind="stable")
    cuts = np.searchsorted(bp[order], np.arange(len(off)))
    rows = []  # (patient, decision row, final row, drug, line) of the current line (the active one with the latest start)
    for i in range(len(off) - 1):
        o = order[cuts[i]:cuts[i + 1]]
        act = o[(bs[o] <= cut) & (be[o] >= cut)]
        if not len(act):
            continue
        j = act[np.argmax(bs[act])]
        s = slice(off[i], off[i + 1])
        dec, fin = np.flatnonzero((kind[s] == DECISION) & (day[s] == bs[j] - 1)), np.flatnonzero(kind[s] == FINAL)
        if len(dec) and len(fin) and d["y_mask"][off[i] + fin[0]] > 0:
            rows.append((i, off[i] + dec[0], off[i] + fin[0], int(bd[j]), int(bl[j])))
    R = np.array(rows)
    pat, dr, fr = R[:, 0], R[:, 1], R[:, 2]
    X = np.where(d["M"][dr] > 0, d["X"][dr], np.nan)[:, keep].astype(np.float32)
    extra = np.column_stack([np.eye(3)[R[:, 3]], R[:, 4], (cut - day[dr]) / MONTH]).astype(np.float32)
    Zi, Zr = PCA(16, random_state=0).fit_transform(d["img_emb"]), PCA(16, random_state=0).fit_transform(d["rna_emb"])
    ir, rr = d["img_row"][dr], d["rna_row"][dr]
    zi = np.where((ir >= 0)[:, None], Zi[np.clip(ir, 0, None)], np.nan).astype(np.float32)
    zr = np.where((rr >= 0)[:, None], Zr[np.clip(rr, 0, None)], np.nan).astype(np.float32)
    y = d["y"][fr]
    pool = np.random.default_rng(0).permutation(np.flatnonzero(d["is_test_patient"] == 0))  # the validation split of hs_e2e.py
    is_val = np.zeros(len(off) - 1, bool)
    is_val[pool[:int(0.1 * len(pool))]] = True
    held = d["is_test_patient"][pat] > 0
    tr, va = ~held & ~is_val[pat], ~held & is_val[pat]
    print(f"rows (open lines with a decision step): {len(R)}; training {tr.sum()}, validation {va.sum()}, held-out {held.sum()}; "
          f"with an RNAseq sample at the decision step: training {(tr & (rr >= 0)).sum()}, held-out {(held & (rr >= 0)).sum()}")
    preds = {}
    for k, Z in {"clin": np.hstack([X, extra]), "img": np.hstack([X, extra, zi]), "full": np.hstack([X, extra, zi, zr])}.items():
        m = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, early_stopping_rounds=100, eval_metric="mae", subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=args.n_jobs, random_state=0, objective="reg:absoluteerror")
        m.fit(Z[tr], y[tr], eval_set=[(Z[va], y[va])], verbose=False)
        preds[k] = m.predict(Z[held])
        print(f"  {k}: best iteration {m.best_iteration}, validation MAE {m.best_score:.3f}", flush=True)
    pid, yt = d["patient_id"][pat[held]], y[held]
    np.savez(f"{args.out}/preds.npz", patient_id=pid, y=yt, **preds)

    g = dict(np.load(f"{args.gru}/heads_preds.npz"))
    gpos = {p: i for i, p in enumerate(g["patient_id"])}
    both = np.array([p in gpos for p in pid])
    gi = np.array([gpos[p] for p in pid[both]])
    yy = yt[both]
    assert np.allclose(yy, g["y"][gi]), "the two files have other labels"
    xc = np.load(f"{args.xgb}/final_preds.npz")
    xpos = {p: i for i, p in enumerate(xc["patient_id"])}
    M = {"GRU head 1 at the cutoff": g["full_h1"][gi], "XGBoost at the cutoff": xc["full_mu"][[xpos[p] for p in pid[both]]]}
    for k, lab in (("clin", "clinical only"), ("img", "+ image"), ("full", "+ image + RNAseq")):
        M[f"GRU head 2 from the start, {lab}"] = g[f"{k}_h2start"][gi]
        M[f"XGBoost from the start, {lab}"] = preds[k][both]
    rng = np.random.default_rng(0)

    def ci(a, b, s):  # mean of |a - y| - |b - y| over the patients of s, 95% interval
        e = np.abs(a[s] - yy[s]) - np.abs(b[s] - yy[s])
        m = e[rng.integers(0, s.sum(), (2000, s.sum()))].mean(1)
        return f"{e.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"

    print(f"\nheld-out patients with an open line: {len(pid)} (XGBoost rows), {len(g['patient_id'])} (GRU), {both.sum()} in both")
    for gname, s in (("all", np.ones(len(yy), bool)), ("with an RNAseq sample", g["has_rna"][gi])):
        print(f"\n{gname}: n={s.sum()}  (MAE against the real label)")
        for k, v in M.items():
            print(f"  {k:46s} {np.mean(np.abs(v[s] - yy[s])):.3f}")
        print(f"  GRU - XGBoost from the start (+ image + RNAseq)   {ci(M['GRU head 2 from the start, + image + RNAseq'], M['XGBoost from the start, + image + RNAseq'], s)}")
        print(f"  RNAseq gain of XGBoost from the start            {ci(M['XGBoost from the start, + image'], M['XGBoost from the start, + image + RNAseq'], s)}")
        print(f"  RNAseq gain of GRU head 2 from the start         {ci(M['GRU head 2 from the start, + image'], M['GRU head 2 from the start, + image + RNAseq'], s)}")


if __name__ == "__main__":
    main()
