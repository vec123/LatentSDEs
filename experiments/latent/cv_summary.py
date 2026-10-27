"""
Summary of the 5-fold cross-validation of the end-to-end GRU (hs_e2e.py --fold f/5 --drop ethnicity; the run folders are listed in
runs/cv5_runs.txt). Every patient of the train folder is held out once; fold 0 is the held-out set of hs_build.py.

1. MAE at the final step (real labels) per fold and group, and pooled over all folds with a bootstrap 95% interval over patients;
   the hidden-modality variants (task 3).
2. Untrained baselines on the same folds (12-month mean + offset, last score + offset; offsets from the training patients of the fold), so that
   the difficulty of a fold is visible, and the ratio GRU / baseline.
3. Coverage of the 90% Laplace interval of head 1 (mu +- b ln 10).
4. Fold 0, the same patients as before: GRU without ethnicity against the tuned XGBoost (with ethnicity) and against the v3 GRU with ethnicity
   (seed 0), paired bootstrap.
5. Test folders: quarter-end derived labels per fold and for the mean of the 5 models.
6. The baselines on the same folds, when their runs exist: tuned XGBoost without ethnicity (xgb_e2e.py --tuned --fold f/5 --drop ethnicity, in
   runs/xgb_e2e_cv5_fold<f>) and the other team's model (cv_other_plan.py; base and with their exercise 3 corrections: 3b for image + RNAseq
   patients, 3a for recent biologic starters with an image). Paired differences pooled over all patients and per fold; means of two models.

    python -I experiments/latent/cv_summary.py [runs/cv5_runs.txt]
"""
import os
import sys

import numpy as np

TRAIN = "test_data/hs_public_dataset/hs_e2e_train.npz"
OLD_XGB, OLD_GRU = "runs/xgb_e2e_tuned", "runs/hs_e2e_20261025-085104"
FINAL = 2
rng = np.random.default_rng(0)
lines = []


def out(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def boot(e, n=2000):
    m = e[rng.integers(0, len(e), (n, len(e)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    listing = sys.argv[1] if len(sys.argv) > 1 else "runs/cv5_runs.txt"
    runs = {}
    for ln in open(listing):
        p = ln.split()
        if len(p) >= 6 and p[0] == "fold" and p[3] == "0":
            runs[int(p[1])] = p[5]
    out(f"folds with a finished run: {sorted(runs)} ({', '.join(runs[f] for f in sorted(runs))})")
    d = np.load(TRAIN)
    names, off, kind = [str(n) for n in d["feature_names"]], d["offsets"], d["kind"]
    n_pat = len(off) - 1
    fr_all = np.array([off[i] + int(np.flatnonzero(kind[off[i]:off[i + 1]] == FINAL)[0]) for i in range(n_pat)])
    col = lambda c: np.where(d["M"][fr_all, names.index(c)] > 0, d["X"][fr_all, names.index(c)], np.nan)
    y_all, lab_all = d["y"][fr_all], d["y_mask"][fr_all] > 0
    has_img, has_rna = d["img_row"][fr_all] >= 0, d["rna_row"][fr_all] >= 0
    groups = {"all patients": np.ones(n_pat, bool), "without visits": col("n_visits_so_far") == 0,
              "open biologic line (task 2)": col("Biologic_active") > 0, "with image (task 3a)": has_img,
              "with image and RNAseq (task 3b)": has_img & has_rna}
    pos = {p: i for i, p in enumerate(d["patient_id"])}

    P = {}  # pooled per patient: fold, mu of the variants, b
    for f, r in sorted(runs.items()):
        g = np.load(f"{r}/final_preds.npz")
        idx = np.array([pos[p] for p in g["patient_id"]])
        assert np.allclose(g["y"], y_all[idx])
        P.setdefault("idx", []).append(idx)
        P.setdefault("fold", []).append(np.full(len(idx), f))
        for k in ("full", "no_img", "no_rna", "clinical"):
            P.setdefault(k, []).append(g[f"{k}_mu"])
        P.setdefault("b", []).append(g["full_b"])
    P = {k: np.concatenate(v) for k, v in P.items()}
    idx, fold = P["idx"], P["fold"]
    y = y_all[idx]
    out(f"pooled: {len(idx)} held-out patients (each once)\n")

    # 1. per fold and pooled
    out("1. MAE at the final step (real labels), GRU without ethnicity")
    out(f"{'group':34s} " + " ".join(f"{'fold ' + str(f):>8s}" for f in sorted(runs)) + f"  {'mean':>6s} {'sd':>5s}   pooled [95%]        n")
    for name, gm in groups.items():
        s = gm[idx]
        per = [np.mean(np.abs(P["full"] - y)[s & (fold == f)]) for f in sorted(runs)]
        e = np.abs(P["full"] - y)[s]
        lo, hi = boot(e)
        out(f"{name:34s} " + " ".join(f"{v:8.3f}" for v in per) + f"  {np.mean(per):6.3f} {np.std(per, ddof=1) if len(per) > 1 else 0:5.3f}   "
            f"{e.mean():.3f} [{lo:.3f}, {hi:.3f}] {s.sum():6d}")
    out("\ntask 3, pooled: MAE with the modalities hidden (one model)")
    out(f"{'group':34s} {'full':>7s} {'no image':>9s} {'no RNA':>7s} {'clinical':>9s}")
    for name in ("with image (task 3a)", "with image and RNAseq (task 3b)"):
        s = groups[name][idx]
        out(f"{name:34s} " + " ".join(f"{np.mean(np.abs(P[k] - y)[s]):{w}.3f}" for k, w in (("full", 7), ("no_img", 9), ("no_rna", 7), ("clinical", 9))))
        for a, b in (("clinical", "no_rna"), ("no_rna", "full")):
            e = np.abs(P[b] - y)[s] - np.abs(P[a] - y)[s]
            lo, hi = boot(e)
            out(f"   gain {a} -> {b}: {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # 2. untrained baselines on the same folds
    out("\n2. untrained baselines on the same folds (offset = median of y - x over the training patients of the fold)")
    base = {"12-month mean + offset": col("mean_12m"), "last score + offset": col("last_score")}
    out(f"{'':34s} " + " ".join(f"{'fold ' + str(f):>8s}" for f in sorted(runs)) + "   pooled")
    ratios = {}
    for bname, x in base.items():
        pred = np.zeros(len(idx))
        for f in sorted(runs):
            te = idx[fold == f]
            tr = np.setdiff1d(np.flatnonzero(lab_all), te)
            c = np.median(y_all[tr])
            ok = tr[np.isfinite(x[tr])]
            o = np.median(y_all[ok] - x[ok])
            pred[fold == f] = np.where(np.isfinite(x[te]), x[te] + o, c)
        e = np.abs(pred - y)
        per = [e[fold == f].mean() for f in sorted(runs)]
        ratios[bname] = [np.mean(np.abs(P["full"] - y)[fold == f]) / v for f, v in zip(sorted(runs), per)]
        out(f"{bname:34s} " + " ".join(f"{v:8.3f}" for v in per) + f"   {e.mean():.3f}")
    for bname, rr in ratios.items():
        out(f"{'GRU / ' + bname:34s} " + " ".join(f"{v:8.3f}" for v in rr) + f"   mean {np.mean(rr):.3f}")

    # 3. calibration
    inside = np.abs(P["full"] - y) <= P["b"] * np.log(10)
    out(f"\n3. coverage of the 90% interval of head 1 (pooled): {inside.mean():.3f}; per fold " +
        ", ".join(f"{inside[fold == f].mean():.3f}" for f in sorted(runs)))

    # 4. fold 0 against the earlier models on the same patients
    if 0 in runs:
        out("\n4. fold 0 (the held-out set used so far): paired differences on the same patients, 95% interval over patients")
        s0 = fold == 0
        g_new, pid0 = P["full"][s0], d["patient_id"][idx[s0]]
        for lab, path in (("tuned XGBoost (with ethnicity)", OLD_XGB), ("GRU v3 seed 0 (with ethnicity)", OLD_GRU)):
            o = np.load(f"{path}/final_preds.npz")
            assert (o["patient_id"] == pid0).all()
            for name, gm in groups.items():
                s = gm[idx[s0]]
                e = np.abs(g_new - y[s0])[s] - np.abs(o["full_mu"] - y[s0])[s]
                lo, hi = boot(e)
                out(f"   GRU without ethnicity - {lab:30s} {name:32s} {np.abs(g_new - y[s0])[s].mean():.3f} vs "
                    f"{np.abs(o['full_mu'] - y[s0])[s].mean():.3f}: {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # 5. test folders
    out("\n5. test folders: MAE at the quarter-end derived labels (the real targets are hidden)")
    for folder in ("test_1", "test_2"):
        preds, rows, yq = [], None, None
        for f, r in sorted(runs.items()):
            sp = np.load(f"{r}/shift_preds.npz")
            if rows is None:
                rows, yq = sp[f"{folder}_rows"], sp[f"{folder}_y"]
            assert (sp[f"{folder}_rows"] == rows).all()
            preds.append(sp[f"{folder}_pred"])
        per = [np.mean(np.abs(p - yq)) for p in preds]
        out(f"   {folder}: per fold " + ", ".join(f"{v:.3f}" for v in per) + f"; mean of the {len(preds)} models {np.mean(np.abs(np.mean(preds, 0) - yq)):.3f} "
            f"(n={len(yq)})")
    # 6. baselines on the same folds
    keyed = {str(p): i for i, p in enumerate(d["patient_id"])}
    B = {}
    xs = [f"runs/xgb_e2e_cv5_fold{f}/final_preds.npz" for f in sorted(runs)]
    if all(os.path.exists(x) for x in xs):
        pr = np.full(n_pat, np.nan)
        for x in xs:
            o = np.load(x)
            pr[[keyed[str(p)] for p in o["patient_id"]]] = o["full_mu"]
        B["XGBoost (no ethnicity)"] = pr[idx]
    if os.path.exists("runs/other_plan/cv5_preds.npz"):
        o = np.load("runs/other_plan/cv5_preds.npz")
        ii = np.array([keyed[str(p)] for p in o["patient_id"]])
        base, p3a, p3b = (np.full(n_pat, np.nan) for _ in range(3))
        base[ii], p3a[ii], p3b[ii] = o["pred"], o["pred_ex3a"], o["pred_ex3b"]
        B["HSBreakers"] = base[idx]
        B["HSBreakers + ex3"] = np.where(p3b != base, p3b, p3a)[idx]  # 3b model for image + RNAseq patients, 3a for recent starters with an image
    if B:
        assert all(np.isfinite(v).all() for v in B.values()), "a baseline misses patients"
        out("\n6. baselines on the same folds: MAE pooled over all held-out patients, and GRU - baseline with a 95% interval over patients")
        out(f"{'group':34s} {'n':>6s} {'GRU':>7s} " + " ".join(f"{k:>22s}" for k in B))
        for name, gm in groups.items():
            s_ = gm[idx]
            out(f"{name:34s} {s_.sum():6d} {np.mean(np.abs(P['full'] - y)[s_]):7.3f} " + " ".join(f"{np.mean(np.abs(v - y)[s_]):22.3f}" for v in B.values()))
        for k, v in B.items():
            for name, gm in groups.items():
                s_ = gm[idx]
                e = np.abs(P["full"] - y)[s_] - np.abs(v - y)[s_]
                lo, hi = boot(e)
                out(f"   GRU - {k:24s} {name:34s} {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if hi < 0 else ("  !" if lo > 0 else "")))
            per = [np.mean(np.abs(P["full"] - y)[fold == f]) - np.mean(np.abs(v - y)[fold == f]) for f in sorted(runs)]
            out(f"   GRU - {k:24s} {'all patients, per fold':34s} " + ", ".join(f"{x:+.3f}" for x in per))
        out("\n   means of two models, pooled (MAE; difference to the GRU alone)")
        for k, v in B.items():
            m2 = (P["full"] + v) / 2
            for name, gm in groups.items():
                s_ = gm[idx]
                e = np.abs(m2 - y)[s_] - np.abs(P["full"] - y)[s_]
                lo, hi = boot(e)
                out(f"   (GRU + {k}) / 2 {name:34s} {np.mean(np.abs(m2 - y)[s_]):.3f}   - GRU {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    open("runs/cv5_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\nsaved runs/cv5_summary.txt")


if __name__ == "__main__":
    main()
