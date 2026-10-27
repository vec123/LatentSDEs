"""
Flares from the latent severity curves of a saved fused model (no training): g(z_t), the severity without visit-type effects (decoder of the
measurement model, posterior mean), at every step of every train-folder patient (visits, quarter ends, cutoff; forward filter, so a step
reads no future data).

Flare (model): a run of consecutive steps with g >= 1.25 x the patient's median g and g - median >= 2 points; per patient the flares per
year of record, the mean duration (days from the first step above to the first step below) and the mean peak excess (points over the median).
Raw-data proxies: flare-reactive visits per year (the patient came for a flare), and flares by the same rule on the raw visit scores.
Associations (all train-folder patients): OLS of each outcome on the static EHR, the share of the record on a biologic / an antibiotic and,
for the RNAseq patients, the first two RNAseq PCs; standardized coefficients with 95% bootstrap intervals over patients. Within-patient
treatment check: patients with steps both on and off a biologic, flare steps per year on vs off (paired). These are associations: the
biologics go to the more severe patients.

    python -m experiments.latent.hs_flares [runs/hs_fused_sde_<stamp>] [--device cuda]
"""
import argparse

import numpy as np
import pandas as pd
import torch

from experiments.latent.hs_e2e import PATHS, VISIT, Data
from experiments.latent.hs_fused_sde import make_model, prepare

DEMO = "test_data/hs_public_dataset/public/train/demographics.csv"
YEAR = 365.25


def curves(model, data, device):
    """g(z_t) in score points at every row"""
    G = np.zeros(len(data.day))
    with torch.no_grad():
        for a in range(0, data.n_pat, 200):
            sel = list(range(a, min(a + 200, data.n_pat)))
            model.encode(data.pack(sel, device))
            mu = model.decode(model.last_post[0])[0].cpu().numpy()
            for j, i in enumerate(sel):
                G[data.off[i]:data.off[i + 1]] = mu[j, :data.n_rows(i)]
    return np.expm1(G * data.stats["ls"] + data.stats["lm"])


def episodes(day, s, rel=1.25, absd=2.0):
    """(number, mean duration in days, mean peak excess) of the flares of one curve (s at days day)"""
    if len(s) < 3:
        return 0, np.nan, np.nan
    base = np.median(s)
    on = (s >= rel * base) & (s - base >= absd)
    n, dur, peak, k = 0, [], [], 0
    while k < len(s):
        if on[k]:
            e = k
            while e + 1 < len(s) and on[e + 1]:
                e += 1
            n += 1
            dur.append((day[e + 1] if e + 1 < len(s) else day[e]) - day[k])
            peak.append(s[k:e + 1].max() - base)
            k = e + 1
        else:
            k += 1
    return n, (np.mean(dur) if dur else np.nan), (np.mean(peak) if peak else np.nan)


def ols_boot(X, y, names, n_boot=300, seed=0):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    fit = lambda A, y: np.linalg.lstsq(A, y, rcond=None)[0][1:]
    b = fit(A, y)
    rng = np.random.default_rng(seed)
    B = np.array([fit(A[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    return {n: (b[k], lo[k], hi[k]) for k, n in enumerate(names)}, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/hs_fused_sde_20261026-190817")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(f"{args.run}/model.pt", map_location=args.device, weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    G = curves(model, data, args.device)
    nm = data.names.index
    demo = pd.read_csv(DEMO).set_index("patient_id")
    rows = []
    for i in range(data.n_pat):
        s = slice(data.off[i], data.off[i + 1])
        day, g = data.day[s].astype(float), G[s]
        years = max((day[-1] - day[0]) / YEAR, 0.25)
        n, dur, peak = episodes(day, g)
        vis = (data.kind[s] == VISIT) & data.has_score[s]
        rn = episodes(day[vis], data.score[s][vis])[0] if vis.sum() >= 3 else np.nan
        r0 = data.off[i]
        rows.append(dict(
            flares_per_year=n / years, duration=dur, peak=peak, raw_flares_per_year=rn / years if np.isfinite(rn) else np.nan,
            flare_visits_per_year=((data.vt[s] == 1).sum()) / years, mean_g=g.mean(),
            bio_share=(data.raw[s, nm("Biologic_active")] > 0).mean(), abx_share=(data.raw[s, nm("Antibiotic_active")] > 0).mean(),
            female=float(demo.loc[data.pid[i], "sex"] == "Female"), black=float(demo.loc[data.pid[i], "ethnicity"] == "Black/African American"),
            bmi=data.raw[r0, nm("bmi_at_baseline")] if data.M[r0, nm("bmi_at_baseline")] else np.nan,
            smoker=float(demo.loc[data.pid[i], "smoking_status"] == "Current"),
            hurley=data.raw[r0, nm("hurley_stage_baseline")] if data.M[r0, nm("hurley_stage_baseline")] else np.nan,
            family=float(demo.loc[data.pid[i], "family_history_hs"] == "Yes"),
            age_dx=data.raw[r0, nm("age_at_diagnosis")] if data.M[r0, nm("age_at_diagnosis")] else np.nan,
            rna=data.rna_row[data.off[i + 1] - 1]))
    P = pd.DataFrame(rows)
    print(f"model {args.run}: {data.n_pat} train-folder patients, {len(G)} steps")
    print("\nA. Flare outcomes (per patient): median [IQR]")
    for c in ("flares_per_year", "duration", "peak", "raw_flares_per_year", "flare_visits_per_year"):
        v = P[c].dropna()
        print(f"   {c:24s} n={len(v):5d}  {v.median():.2f} [{v.quantile(.25):.2f}, {v.quantile(.75):.2f}]  mean {v.mean():.2f}")
    sp = P[["flares_per_year", "raw_flares_per_year", "flare_visits_per_year"]].corr(method="spearman")
    print(f"   Spearman: model flares vs raw-score flares {sp.iloc[0, 1]:+.2f}, model flares vs flare-reactive visits {sp.iloc[0, 2]:+.2f}, "
          f"raw-score flares vs flare-reactive visits {sp.iloc[1, 2]:+.2f}")

    covs = ["bio_share", "abx_share", "female", "black", "bmi", "smoker", "hurley", "family", "age_dx"]
    print("\nB. Associations, all patients: standardized OLS coefficient [95% bootstrap] (outcome units per 1 SD of the covariate)")
    for out in ("flares_per_year", "duration", "peak", "mean_g", "raw_flares_per_year", "flare_visits_per_year"):
        res, n = ols_boot(P[covs].to_numpy(float), P[out].to_numpy(float), covs)
        print(f"   {out} (n={n}): " + "; ".join(f"{k} {b:+.3f} [{lo:+.3f}, {hi:+.3f}]" for k, (b, lo, hi) in res.items()))

    # RNAseq PCs (standardized scores of the sample at the cutoff), RNAseq patients only
    E = data.rna_emb.astype(np.float64)
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    sc = U[:, :2] / U[:, :2].std(0)
    R = P[P.rna >= 0].copy()
    R["rna_pc1"], R["rna_pc2"] = sc[R.rna.to_numpy(int), 0], sc[R.rna.to_numpy(int), 1]
    covr = ["rna_pc1", "rna_pc2"] + covs[1:]
    print(f"\nC. RNAseq patients (n={len(R)}; all start a biologic at the sample): standardized OLS coefficient [95% bootstrap]")
    for out in ("flares_per_year", "peak", "mean_g", "raw_flares_per_year", "flare_visits_per_year"):
        res, n = ols_boot(R[covr].to_numpy(float), R[out].to_numpy(float), covr)
        print(f"   {out} (n={n}): " + "; ".join(f"{k} {b:+.3f} [{lo:+.3f}, {hi:+.3f}]" for k, (b, lo, hi) in res.items() if k.startswith("rna")))

    # within patient: flare steps per year on vs off a biologic
    on_r, off_r = [], []
    for i in range(data.n_pat):
        s = slice(data.off[i], data.off[i + 1])
        day, g = data.day[s].astype(float), G[s]
        if len(g) < 6:
            continue
        base = np.median(g)
        fl = (g >= 1.25 * base) & (g - base >= 2)
        bio = data.raw[s, nm("Biologic_active")] > 0
        dt = np.diff(day, append=day[-1])
        if dt[bio].sum() > 180 and dt[~bio].sum() > 180:
            on_r.append(fl[bio].sum() / (dt[bio].sum() / YEAR))
            off_r.append(fl[~bio].sum() / (dt[~bio].sum() / YEAR))
    on_r, off_r = np.array(on_r), np.array(off_r)
    d = on_r - off_r
    B = np.array([d[np.random.default_rng(k).integers(0, len(d), len(d))].mean() for k in range(500)])
    print(f"\nD. Within patient (n={len(d)}, > 180 days both on and off a biologic): flare steps per year on {on_r.mean():.2f}, off "
          f"{off_r.mean():.2f}; on - off {d.mean():+.2f} [{np.percentile(B, 2.5):+.2f}, {np.percentile(B, 97.5):+.2f}]")


if __name__ == "__main__":
    main()
