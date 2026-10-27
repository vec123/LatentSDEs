"""
Ethnicity in the three folders and in the GRU (CPU, under a minute; no training).

A. Mix of ethnicity and the other demographics per folder (train, test_1, test_2).
B. Train folder: the real target and the residual of an untrained baseline (12-month mean + offset) per ethnicity: does ethnicity tell something
   about the next 6 months beyond the patient's own scores? Also the mean visit score per ethnicity in each folder (is the relation the same in
   the test folders?).
C. GRU with ethnicity (--with) and without (--without), the same patients: MAE and mean residual per ethnicity on the held-out patients of the
   train folder (when both runs hold out the same patients) and at the quarter-end derived labels of the test folders, paired bootstrap over
   patients.

    python -I experiments/latent/ethnicity_shift.py --with runs/hs_e2e_<stamp> --without runs/hs_e2e_<stamp>
"""
import argparse

import numpy as np
import pandas as pd

ROOT = "test_data/hs_public_dataset/"
FOLDERS = ("train", "test_1", "test_2")
rng = np.random.default_rng(0)


def boot(e, n=2000):
    if len(e) < 2:
        return np.nan, np.nan
    m = e[rng.integers(0, len(e), (n, len(e)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def patient_boot(e, pat, n=2000):
    """95% interval of the mean of e (rows), resampling patients"""
    u, inv = np.unique(pat, return_inverse=True)
    es, cnt = np.bincount(inv, e), np.bincount(inv)
    j = rng.integers(0, len(u), (n, len(u)))
    return tuple(np.percentile(es[j].sum(1) / cnt[j].sum(1), [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with", dest="with_", default="runs/hs_e2e_20261025-085104")
    ap.add_argument("--without", default="runs/hs_e2e_20261025-131650")
    args = ap.parse_args()
    demo = {f: pd.read_csv(f"{ROOT}public/{f}/demographics.csv").set_index("patient_id") for f in FOLDERS}
    eth = {f: demo[f].ethnicity.fillna("missing") for f in FOLDERS}
    cats = ["White", "Black/African American", "Hispanic/Latino", "Asian", "Other/Unknown", "missing"]

    print("A. ethnicity per folder (share of patients)")
    print(f"{'':24s}" + "".join(f"{f:>10s}" for f in FOLDERS))
    for c in cats:
        print(f"{c:24s}" + "".join(f"{(eth[f] == c).mean():10.3f}" for f in FOLDERS))
    print(f"{'patients':24s}" + "".join(f"{len(eth[f]):10d}" for f in FOLDERS))
    print("\n   other demographics (mean or share)")
    rows = {"female": lambda x: (x.sex == "Female").mean(), "birth year": lambda x: x.birth_year.mean(), "BMI": lambda x: x.bmi_at_baseline.mean(),
            "Hurley stage": lambda x: x.hurley_stage_baseline.mean(), "current smoker": lambda x: (x.smoking_status == "Current").mean(),
            "family history yes": lambda x: (x.family_history_hs == "Yes").mean(), "diagnostic delay (y)": lambda x: x.diagnostic_delay_years.mean()}
    for k, fn in rows.items():
        print(f"   {k:21s}" + "".join(f"{fn(demo[f]):10.2f}" for f in FOLDERS))

    # B. train folder: target and baseline residual per ethnicity; visit scores per folder
    d = np.load(f"{ROOT}hs_e2e_train.npz")
    names, off, kind = [str(n) for n in d["feature_names"]], d["offsets"], d["kind"]
    fr = np.array([off[i] + int(np.flatnonzero(kind[off[i]:off[i + 1]] == 2)[0]) for i in range(len(off) - 1)])
    y, lab = d["y"][fr], d["y_mask"][fr] > 0
    m12 = np.where(d["M"][fr, names.index("mean_12m")] > 0, d["X"][fr, names.index("mean_12m")], np.nan)
    e_tr = eth["train"].reindex(d["patient_id"]).to_numpy()
    ok = lab & np.isfinite(m12)
    resid = y - (m12 + np.median(y[ok] - m12[ok]))
    print("\nB. train folder, per ethnicity: real target (mean), residual of 12-month mean + offset (mean, 95% interval), visit score per folder")
    vis = {f: pd.read_csv(f"{ROOT}public/{f}/visits.csv", usecols=["patient_id", "ihs4_score"]).dropna() for f in FOLDERS}
    print(f"{'':24s}{'n':>6s}{'target':>8s}   {'baseline residual':>24s}" + "".join(f"{'visits ' + f:>14s}" for f in FOLDERS))
    for c in cats:
        s = e_tr == c
        lo, hi = boot(resid[s & ok])
        vm = [vis[f].ihs4_score[vis[f].patient_id.map(eth[f]) == c].mean() for f in FOLDERS]
        print(f"{c:24s}{(s & lab).sum():6d}{y[s & lab].mean():8.2f}   {resid[s & ok].mean():+8.3f} [{lo:+.3f}, {hi:+.3f}]" + "".join(f"{v:14.2f}" for v in vm))

    # C. the two GRU runs
    print(f"\nC. GRU with ethnicity ({args.with_}) against without ({args.without}), the same patients; MAE and mean residual (prediction - label)")
    gw, go = np.load(f"{args.with_}/final_preds.npz"), np.load(f"{args.without}/final_preds.npz")
    if len(gw["patient_id"]) == len(go["patient_id"]) and (gw["patient_id"] == go["patient_id"]).all():
        e_te = eth["train"].reindex(gw["patient_id"]).to_numpy()
        yt = gw["y"]
        print("   held-out patients of the train folder (real labels)")
        for c in ["all"] + cats:
            s = np.ones(len(yt), bool) if c == "all" else e_te == c
            if s.sum() < 5:
                continue
            aw, ao = np.abs(gw["full_mu"] - yt)[s], np.abs(go["full_mu"] - yt)[s]
            lo, hi = boot(ao - aw)
            print(f"   {c:24s} n={s.sum():5d}  MAE with {aw.mean():.3f}, without {ao.mean():.3f}, without - with {np.mean(ao - aw):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"   bias with {np.mean((gw['full_mu'] - yt)[s]):+.3f}, without {np.mean((go['full_mu'] - yt)[s]):+.3f}")
    else:
        print("   held-out patients of the train folder: the runs hold out other patients, no paired comparison")
    sw, so = np.load(f"{args.with_}/shift_preds.npz"), np.load(f"{args.without}/shift_preds.npz")
    for f in ("test_1", "test_2"):
        assert (sw[f"{f}_rows"] == so[f"{f}_rows"]).all()
        t = np.load(f"{ROOT}hs_e2e_{f}.npz")
        pat = sw[f"{f}_patient"]
        e_q = eth[f].reindex(t["patient_id"][pat]).to_numpy()
        yq = sw[f"{f}_y"]
        print(f"   {f}: quarter-end derived labels (rows of the visits; interval over patients)")
        for c in ["all"] + cats:
            s = np.ones(len(yq), bool) if c == "all" else e_q == c
            if s.sum() < 20:
                continue
            aw, ao = np.abs(sw[f"{f}_pred"] - yq)[s], np.abs(so[f"{f}_pred"] - yq)[s]
            lo, hi = patient_boot(ao - aw, pat[s])
            print(f"   {c:24s} n={s.sum():6d} ({len(np.unique(pat[s])):5d} patients)  MAE with {aw.mean():.3f}, without {ao.mean():.3f}, "
                  f"without - with {np.mean(ao - aw):+.3f} [{lo:+.3f}, {hi:+.3f}]   bias with {np.mean((sw[f'{f}_pred'] - yq)[s]):+.3f}, "
                  f"without {np.mean((so[f'{f}_pred'] - yq)[s]):+.3f}")


if __name__ == "__main__":
    main()
