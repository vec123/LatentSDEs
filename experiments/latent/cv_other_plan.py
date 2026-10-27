"""
The other team's model (other_plan/round1 (1).py, "HSBreakers") on the 5 cross-validation folds of hs_e2e.py --fold f/5 (CPU, about 2-3 minutes per
fold). Per fold, as compare_other_plan.py does on fold 0: their feature table (label-free; ethnicity and practitioner id dropped, as they do), their
gradient boosting fitted on the other 4 folds, and their exercise 3 corrections (PCA + ridge on the residual of out-of-fold predictions of the
training patients, applied only if the cross-validated gain has a 95% interval above 0): 3b image + RNAseq stacked, 3a recent biologic starters with an
image. Writes runs/other_plan/cv5_preds.npz (every train-folder patient once: fold, base prediction, 3a and 3b predictions).

    python -I experiments/latent/cv_other_plan.py
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "experiments/latent")
from xgb_e2e import held_out  # noqa: E402  (the folds of hs_e2e.py)

ROOT = Path("test_data/hs_public_dataset/public")
OUT = Path("runs/other_plan")
spec = importlib.util.spec_from_file_location("hsb", "other_plan/round1 (1).py")
hsb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsb)


def fold_predictions(X, y, feats, EMB, tr_ids, te_ids):
    """their model fitted on tr_ids: base prediction of te_ids and the 3a / 3b corrected predictions"""
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5.0,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, categorical_features="from_dtype", random_state=0)
    m.fit(X.loc[tr_ids, feats], y.loc[tr_ids])
    base = pd.Series(np.clip(m.predict(X.loc[te_ids, feats]), 0, None), te_ids)
    oof = pd.Series(hsb.cv_lgb(X.loc[tr_ids, feats], y.loc[tr_ids])[0], tr_ids)
    ic = [c for c in EMB["img"].columns if c.startswith("image_dim_")]
    # 3b: image + RNAseq stacked
    both = EMB["img"].index.intersection(EMB["rna"].index)
    itr, ite = both.intersection(pd.Index(tr_ids)), both.intersection(pd.Index(te_ids))
    Etr, Ete = hsb.stack_embeddings([EMB["img"].loc[itr, ic].values, EMB["rna"].loc[itr, EMB["rna_dims"]].values],
                                    [EMB["img"].loc[ite, ic].values, EMB["rna"].loc[ite, EMB["rna_dims"]].values])
    gain, gci = hsb.emb_gain(y.loc[itr].values, oof.loc[itr].values, Etr)
    p3b = base.copy()
    if gci[0] > 0:
        p3b.loc[ite] = np.clip(base.loc[ite].values + hsb.embedding_correction(Etr, y.loc[itr].values - oof.loc[itr].values, Ete), 0, None)
    # 3a: recent biologic starters with an image (first third of days on the biologic)
    iim = EMB["img"].index
    itr_i, ite_i = iim.intersection(pd.Index(tr_ids)), iim.intersection(pd.Index(te_ids))
    on, bd = X.loc[itr_i, "on_bio"].values == 1, X.loc[itr_i, "bio_days"].values
    thr = np.nanquantile(bd[on], 1 / 3)
    rec = itr_i[on & (bd <= thr)]
    g3a, c3a = hsb.emb_gain(y.loc[rec].values, oof.loc[rec].values, EMB["img"].loc[rec, ic].values)
    r_te = ite_i[(X.loc[ite_i, "on_bio"].values == 1) & (X.loc[ite_i, "bio_days"].values <= thr)]
    p3a = base.copy()
    if c3a[0] > 0 and len(r_te):
        p3a.loc[r_te] = np.clip(base.loc[r_te].values + hsb.embedding_correction(EMB["img"].loc[rec, ic].values, y.loc[rec].values - oof.loc[rec].values,
                                                                                    EMB["img"].loc[r_te, ic].values), 0, None)
    info = (f"{m.n_iter_} iterations; 3b gain {gain:+.3f} [{gci[0]:+.3f}, {gci[1]:+.3f}] applied {gci[0] > 0}; "
            f"3a gain {g3a:+.3f} [{c3a[0]:+.3f}, {c3a[1]:+.3f}] applied {c3a[0] > 0} ({len(r_te)} held-out patients)")
    return base, p3a, p3b, info


def main():
    t0 = time.time()
    ALL = pd.read_pickle(OUT / "features_all.pkl")  # written by compare_other_plan.py
    X = ALL[ALL.split == "train"].drop(columns="split")
    y = pd.read_csv(ROOT / "train" / "targets.csv").set_index("patient_id").avg_ihs4_next6.reindex(X.index)
    feats = [c for c in X.columns if c not in ("practitioner", "ethnicity")]
    EMB = hsb.load_embeddings(ROOT / "train")
    d = np.load("test_data/hs_public_dataset/hs_e2e_train.npz")
    pid, flag0 = d["patient_id"].astype(str), d["is_test_patient"] > 0
    res = {k: [] for k in ("patient_id", "fold", "pred", "pred_ex3a", "pred_ex3b")}
    for f in range(5):
        te = held_out(flag0, 0, (f, 5))
        tr_ids, te_ids = pid[~te], pid[te]
        base, p3a, p3b, info = fold_predictions(X, y, feats, EMB, tr_ids, te_ids)
        print(f"fold {f}: {len(tr_ids)} training, {len(te_ids)} held-out patients; MAE {np.abs(base.values - y.loc[te_ids].values).mean():.3f}; "
              f"{info} ({time.time() - t0:.0f}s)", flush=True)
        for k, v in (("patient_id", te_ids), ("fold", np.full(len(te_ids), f)), ("pred", base.values), ("pred_ex3a", p3a.values), ("pred_ex3b", p3b.values)):
            res[k].append(v)
    np.savez(OUT / "cv5_preds.npz", **{k: np.concatenate(v) for k, v in res.items()})
    print(f"saved {OUT / 'cv5_preds.npz'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
