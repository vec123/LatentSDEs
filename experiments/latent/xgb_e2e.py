"""
XGBoost on the steps of hs_build.py: the baseline of the end-to-end model (hs_e2e.py), with the same information.

Rows: the labelled steps (visits and quarter ends with a derived label, final steps with the real label) of the training patients
(the patients of the train folder that are not held out). Features: the step features (missing = NaN) and 16 principal components of
the image and of the RNAseq embedding from the sample date on (PCA fitted on the train folder). Weights as in hs_e2e.py: per patient
1/sqrt(n) for each kind of derived label, 1 for the real label. L1 objective, the tree settings of xgb_visit_forecast.py.
Scored like hs_e2e.py: the final steps of the held-out patients, also with the image and the RNAseq hidden (set missing), and the
quarter-end derived labels. Writes <out>/final_preds.npz and <out>/model.json.

    python -I experiments/latent/xgb_e2e.py [--out runs/xgb_e2e] [--variants]
"""
import argparse
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

TRAIN = "test_data/hs_public_dataset/hs_e2e_train.npz"
VISIT, QUERY, FINAL, DECISION = 0, 1, 2, 3


def held_out(flag, split, fold=None):
    """The held-out patients (as hs_e2e.held_out): split 0 the is_test_patient flag; split k > 0 a new set of the same size, drawn
    (seed k) from the other patients; fold (f, K): fold 0 the is_test_patient set, folds 1..K-1 a random split (seed 0) of the others."""
    if fold is not None:
        f, k = fold
        if f == 0:
            return flag
        new = np.zeros_like(flag)
        new[np.array_split(np.random.default_rng(0).permutation(np.flatnonzero(~flag)), k - 1)[f - 1]] = True
        return new
    if split == 0:
        return flag
    new = np.zeros_like(flag)
    new[np.random.default_rng(split).choice(np.flatnonzero(~flag), int(flag.sum()), replace=False)] = True
    return new


def design(d, X, Zi, Zr, rows, hide_img=False, hide_rna=False):
    ir, rr = d["img_row"][rows], d["rna_row"][rows]
    zi = np.where(((ir >= 0) & (not hide_img))[:, None], Zi[np.clip(ir, 0, None)], np.nan)
    zr = np.where(((rr >= 0) & (not hide_rna))[:, None], Zr[np.clip(rr, 0, None)], np.nan)
    return np.concatenate([X[rows], zi, zr], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/xgb_e2e")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--variants", action="store_true", help="also train the clinical-only and the clinical + image models (task 3)")
    ap.add_argument("--tests", action="store_true", help="also predict the final steps of test_1 and test_2 (test_preds.npz)")
    ap.add_argument("--skip_clinical", action="store_true", help="keep the saved predictions of the clinical model (no new fit)")
    ap.add_argument("--tuned", action="store_true", help="early stopping on the validation patients of hs_e2e.py (10%% of the training "
                    "patients, seed 0; their rows leave the training rows, as in hs_e2e.py): learning rate 0.05, up to 2,000 trees")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--split", type=int, default=0, help="the held-out set of hs_e2e.py --split")
    ap.add_argument("--fold", default="", help="cross-validation fold f/K of hs_e2e.py --fold")
    ap.add_argument("--drop", default="", help="features removed from the input, comma separated (as hs_e2e.py --drop)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    old_clin = None  # the saved predictions of the clinical model (--skip_clinical), read before this run writes the file again
    if args.skip_clinical and os.path.exists(f"{args.out}/final_preds.npz"):
        o = np.load(f"{args.out}/final_preds.npz")
        if "clinical_model_mu" in o.files:
            old_clin = (o["patient_id"], o["clinical_model_mu"])
    d = np.load(TRAIN)
    names = [str(n) for n in d["feature_names"]]
    drop = {"is_decision"} | {c for c in args.drop.split(",") if c}
    keep = [j for j, n in enumerate(names) if n not in drop]
    names = [names[j] for j in keep]
    X = np.where(d["M"] > 0, d["X"], np.nan)[:, keep].astype(np.float32)
    kind, y, ym, off = d["kind"], d["y"], d["y_mask"] > 0, d["offsets"]
    n_pat = len(d["patient_id"])
    prow = np.repeat(np.arange(n_pat), np.diff(off))
    flag = held_out(d["is_test_patient"] > 0, args.split, tuple(int(v) for v in args.fold.split("/")) if args.fold else None)
    held = flag[prow]
    Zi = PCA(16, random_state=0).fit_transform(d["img_emb"])
    Zr = PCA(16, random_state=0).fit_transform(d["rna_emb"])
    w = np.zeros(len(y))
    for kd in (VISIT, QUERY):
        lab = ym & (kind == kd)
        cnt = np.bincount(prow[lab], minlength=n_pat)
        w[lab] = 1.0 / np.sqrt(cnt[prow[lab]])
    w[ym & (kind == FINAL)] = 1.0
    final_of = lambda pats: np.array([off[i] + int(np.flatnonzero(kind[off[i]:off[i + 1]] == FINAL)[0]) for i in pats])
    base = dict(subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=args.n_jobs, random_state=0, objective="reg:absoluteerror")
    if args.tuned:
        pool = np.random.default_rng(0).permutation(np.flatnonzero(~flag))  # the validation split of hs_e2e.py
        is_val = np.zeros(n_pat, bool)
        is_val[pool[:int(0.1 * len(pool))]] = True
        tr, vfr = np.flatnonzero(ym & ~held & ~is_val[prow]), final_of(np.flatnonzero(is_val))
        params = dict(base, n_estimators=2000, learning_rate=0.05, max_depth=args.depth, early_stopping_rounds=100, eval_metric="mae")
    else:
        tr, vfr = np.flatnonzero(ym & ~held), None
        params = dict(base, n_estimators=300, learning_rate=0.08, max_depth=6)

    def fit(hide_img=False, hide_rna=False):
        m = XGBRegressor(**params)
        kw = {"eval_set": [(design(d, X, Zi, Zr, vfr, hide_img, hide_rna), y[vfr])], "verbose": False} if args.tuned else {}
        m.fit(design(d, X, Zi, Zr, tr, hide_img, hide_rna), y[tr], sample_weight=w[tr] / w[tr].mean(), **kw)
        if args.tuned:
            print(f"  early stopping: best iteration {m.best_iteration}, validation MAE {m.best_score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return m

    model = fit()
    model.save_model(f"{args.out}/model.json")
    print(f"trained on {len(tr)} rows ({(kind[tr] == VISIT).sum()} visits, {(kind[tr] == QUERY).sum()} quarter ends, "
          f"{(kind[tr] == FINAL).sum()} final), {X.shape[1] + 32} features (split {args.split}, removed: {sorted(drop)}), in {time.time() - t0:.0f}s", flush=True)

    te = np.flatnonzero(flag)
    fr = final_of(te)
    yt = y[fr]
    var = {k: model.predict(design(d, X, Zi, Zr, fr, hide_img=hi, hide_rna=hr)) for k, (hi, hr) in
           {"full": (False, False), "no_img": (True, False), "no_rna": (False, True), "clinical": (True, True)}.items()}
    col = names.index
    raw = lambda c: X[fr, col(c)]
    has_img, has_rna = d["img_row"][fr] >= 0, d["rna_row"][fr] >= 0
    gap = np.nan_to_num(raw("days_since_prev_visit"), nan=np.inf)
    groups = {"all held-out patients": np.ones(len(te), bool), "without visits": raw("n_visits_so_far") == 0,
              "last visit > 90 days before": gap > 90, "open biologic line (task 2)": raw("Biologic_active") > 0,
              "with image (task 3a)": has_img, "with image and RNAseq (task 3b)": has_img & has_rna}
    print(f"\nheld-out patients: MAE of XGBoost at the final step (real labels)")
    print(f"{'group':34s} {'n':>5s} {'full':>7s} {'no image':>9s} {'no RNA':>7s} {'clinical':>9s}")
    for g, s in groups.items():
        r = {k: float(np.mean(np.abs(v[s] - yt[s]))) for k, v in var.items()}
        print(f"{g:34s} {s.sum():5d} {r['full']:7.3f} {r['no_img']:9.3f} {r['no_rna']:7.3f} {r['clinical']:9.3f}")
    q = np.flatnonzero(held & (kind == QUERY) & ym)
    print(f"quarter-end derived labels of the held-out patients: n={len(q)} MAE {np.mean(np.abs(model.predict(design(d, X, Zi, Zr, q)) - y[q])):.3f}")
    models = {"full": model}
    save = lambda: np.savez(f"{args.out}/final_preds.npz", patient_id=d["patient_id"][te], y=yt, **{f"{k}_mu": v for k, v in var.items()})
    save()

    def variant(name, hi, hr):  # task 3 as the deck asks: a separately trained model
        m = fit(hi, hr)
        p_ = m.predict(design(d, X, Zi, Zr, fr, hide_img=hi, hide_rna=hr))
        var[name.replace(" ", "_").replace("+", "and")] = p_
        models[name] = m
        print(f"  {name:24s} " + "  ".join(f"{g}: {np.mean(np.abs(p_[groups[g]] - yt[groups[g]])):.3f}" for g in
                                           ("all held-out patients", "with image (task 3a)", "with image and RNAseq (task 3b)")), flush=True)
        save()

    if args.variants or args.tests:
        print(f"\nseparately trained models (task 3): MAE at the final step")
        variant("clinical + image model", False, True)  # first: the answer files need it
    if args.tests:  # the final steps of the two test folders (the PCA of the train folder is applied to their embeddings)
        pca_i, pca_r = PCA(16, random_state=0).fit(d["img_emb"]), PCA(16, random_state=0).fit(d["rna_emb"])
        out = {}
        for folder in ("test_1", "test_2"):
            t = dict(np.load(TRAIN.replace("train", folder)))  # loaded once (an npz decompresses an array at every access)
            Xt = np.where(t["M"] > 0, t["X"], np.nan)[:, keep].astype(np.float32)
            ot, kt = t["offsets"], t["kind"]
            frt = np.array([ot[i] + int(np.flatnonzero(kt[ot[i]:ot[i + 1]] == FINAL)[0]) for i in range(len(t["patient_id"]))])
            Zit, Zrt = pca_i.transform(t["img_emb"]), pca_r.transform(t["rna_emb"])
            q = np.flatnonzero((kt == QUERY) & (t["y_mask"] > 0))  # the quarter-end derived labels (from the test visits)
            out[folder] = {"patient_id": t["patient_id"], "full": models["full"].predict(design(t, Xt, Zit, Zrt, frt)),
                           "image": models["clinical + image model"].predict(design(t, Xt, Zit, Zrt, frt, hide_rna=True)),
                           "quarter_rows": q, "quarter_pred": models["full"].predict(design(t, Xt, Zit, Zrt, q)), "quarter_y": t["y"][q]}
            print(f"{folder}: quarter-end derived labels n={len(q)} MAE {np.mean(np.abs(out[folder]['quarter_pred'] - t['y'][q])):.3f}", flush=True)
        np.savez(f"{args.out}/test_preds.npz", **{f"{f}_{k}": v for f, o in out.items() for k, v in o.items()})
        print(f"saved {args.out}/test_preds.npz (test_1 and test_2: final steps of the full and the clinical + image model, quarter ends)",
              flush=True)
    if args.variants:
        if old_clin is not None and (old_clin[0] == d["patient_id"][te]).all():
            var["clinical_model"] = old_clin[1]
            print("  clinical model: the saved predictions are kept")
            save()
        else:
            variant("clinical model", True, True)
    print(f"saved {args.out}/final_preds.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
