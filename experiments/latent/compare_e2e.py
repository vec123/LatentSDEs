"""
Paired comparison of the end-to-end GRU (hs_e2e.py) and XGBoost (xgb_e2e.py) on the same held-out patients.

MAE at the final step (real labels) per group, with a bootstrap over patients (1,000 draws) of the difference GRU - XGBoost;
also the mean of the two models (an ensemble). Task 3: the GRU is one model with the modalities hidden; XGBoost has separately
trained models (clinical only; clinical + image) and its full model.

    python -I experiments/latent/compare_e2e.py runs/hs_e2e_<stamp> [runs/xgb_e2e]
"""
import sys

import numpy as np

TRAIN = "test_data/hs_public_dataset/hs_e2e_train.npz"
FINAL = 2


def main():
    g = np.load(f"{sys.argv[1]}/final_preds.npz")
    x = np.load(f"{sys.argv[2] if len(sys.argv) > 2 else 'runs/xgb_e2e'}/final_preds.npz")
    assert (g["patient_id"] == x["patient_id"]).all() and np.allclose(g["y"], x["y"]), "the two runs have other patients"
    d = np.load(TRAIN)
    names, off, kind = [str(n) for n in d["feature_names"]], d["offsets"], d["kind"]
    pos = {p: i for i, p in enumerate(d["patient_id"])}
    fr = np.array([off[pos[p]] + int(np.flatnonzero(kind[off[pos[p]]:off[pos[p] + 1]] == FINAL)[0]) for p in g["patient_id"]])
    raw = lambda c: np.where(d["M"][fr, names.index(c)] > 0, d["X"][fr, names.index(c)], np.nan)
    has_img, has_rna = d["img_row"][fr] >= 0, d["rna_row"][fr] >= 0
    groups = {"all held-out patients": np.ones(len(fr), bool), "with visits": raw("n_visits_so_far") > 0,
              "without visits": raw("n_visits_so_far") == 0, "last visit > 90 days before": np.nan_to_num(raw("days_since_prev_visit"), nan=1e9) > 90,
              "open biologic line (task 2)": raw("Biologic_active") > 0, "with image (task 3a)": has_img,
              "with image and RNAseq (task 3b)": has_img & has_rna}
    y = g["y"]
    rng = np.random.default_rng(0)

    def compare(a, b, s):
        e = np.abs(a[s] - y[s]) - np.abs(b[s] - y[s])
        m = e[rng.integers(0, s.sum(), (1000, s.sum()))].mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        return f"{np.mean(np.abs(a[s] - y[s])):6.3f} {np.mean(np.abs(b[s] - y[s])):7.3f}   {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]" + \
               ("  *" if hi < 0 else ("  !" if lo > 0 else ""))

    ens = 0.5 * (g["full_mu"] + x["full_mu"])
    print("MAE at the final step; difference with a 95% interval over patients; * the first model is better, ! the second\n")
    print(f"{'group':34s} {'n':>5s}   {'GRU':>6s} {'XGBoost':>7s}   GRU - XGBoost")
    for name, s in groups.items():
        print(f"{name:34s} {s.sum():5d}   {compare(g['full_mu'], x['full_mu'], s)}")
    print(f"\n{'group':34s} {'n':>5s}   {'mean':>6s} {'XGBoost':>7s}   (GRU + XGBoost) / 2 - XGBoost")
    for name, s in groups.items():
        print(f"{name:34s} {s.sum():5d}   {compare(ens, x['full_mu'], s)}")
    print("\ntask 3: gain of the modalities within each model (MAE: clinical only -> + image -> + image + RNAseq)")
    gx = {"clinical": x["clinical_model_mu"] if "clinical_model_mu" in x.files else x["clinical_mu"],
          "image": x["clinical_and_image_model_mu"] if "clinical_and_image_model_mu" in x.files else x["no_rna_mu"], "full": x["full_mu"]}
    gg = {"clinical": g["clinical_mu"], "image": g["no_rna_mu"], "full": g["full_mu"]}
    for name in ("with image (task 3a)", "with image and RNAseq (task 3b)"):
        s = groups[name]
        for lab, P in (("GRU (one model, modalities hidden)", gg), ("XGBoost (separate models)", gx)):
            c, i_, f_ = (float(np.mean(np.abs(P[k][s] - y[s]))) for k in ("clinical", "image", "full"))
            print(f"  {name:32s} {lab:36s} {c:.3f} -> {i_:.3f} -> {f_:.3f}   gain of image {100 * (c - i_) / c:+.1f}%, of both {100 * (c - f_) / c:+.1f}%")
        print(f"  {name:32s} {'GRU - XGBoost, full models':36s} {compare(g['full_mu'], x['full_mu'], s)}")
    # the test folders: quarter-end derived labels from their own visits (test_2 is the shifted population); bootstrap over patients
    xdir = sys.argv[2] if len(sys.argv) > 2 else "runs/xgb_e2e"
    try:
        sg, sx = np.load(f"{sys.argv[1]}/shift_preds.npz"), np.load(f"{xdir}/test_preds.npz")
    except FileNotFoundError:
        return
    print("\ntest folders: MAE at the quarter-end derived labels (the real targets of the test folders are not available)")
    for folder in ("test_1", "test_2"):
        if f"{folder}_quarter_rows" not in sx.files:
            continue
        assert (sg[f"{folder}_rows"] == sx[f"{folder}_quarter_rows"]).all()
        yq, pg, px, pat = sg[f"{folder}_y"], sg[f"{folder}_pred"], sx[f"{folder}_quarter_pred"], sg[f"{folder}_patient"]
        e = np.abs(pg - yq) - np.abs(px - yq)
        u, inv = np.unique(pat, return_inverse=True)
        es, cnt = np.bincount(inv, e), np.bincount(inv)
        j = rng.integers(0, len(u), (1000, len(u)))
        lo, hi = np.percentile(es[j].sum(1) / cnt[j].sum(1), [2.5, 97.5])
        print(f"  {folder}: n={len(yq)} steps of {len(u)} patients  GRU {np.mean(np.abs(pg - yq)):.3f}  XGBoost {np.mean(np.abs(px - yq)):.3f}"
              f"  GRU - XGBoost {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
