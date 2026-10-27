"""
Figure: the first principal component (PC1) of the baseline RNAseq embedding and the response to the biologic (CPU, under 1 minute).

Starts and response as rna_profiles.py: the biologic starts with an RNAseq sample on the start day, ratio = mean IHS4 days 90..270
after / mean 180 days before (pre mean >= 3); good responder: ratio < 0.5. PCA on all 600 RNAseq samples (no label), components
scaled to unit variance. The files hold the 128-d embedding only (no genes).

A. ratio against PC1 (one point per start; marker = drug), median per fifth of PC1 with a bootstrap 95% interval;
   Spearman correlation, p from 10,000 shuffles, also against the largest |correlation| of the first 20 components (selection).
B. mean IHS4 relative to the 6 months before the start, per 2-month bin, for the thirds of PC1 (bootstrap 95% band over starts).
C. share of good responders per third (Wilson 95% interval); out-of-fold AUC (5-fold x 5) of a logistic model on the clinical
   inputs, and on the clinical inputs + the component with the largest difference in the training fold.

    python -I experiments/latent/rna_pc1_plot.py [--out runs/rna_pc1] [--columns <png>]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TRAIN = "test_data/hs_public_dataset/public/train/"
C_LOW, C_MID, C_HIGH, C_INK, C_GRID = "#2b6cb0", "#9a9a9a", "#c05621", "#111111", "#e6e6e6"
MARK = {"Adalimumab": "o", "Bimekizumab": "s", "Secukinumab": "^"}


def starts():
    t = pd.read_csv(TRAIN + "treatments.csv", parse_dates=["start_date"])
    v = pd.read_csv(TRAIN + "visits.csv", parse_dates=["visit_date"]).dropna(subset=["ihs4_score"])
    demo = pd.read_csv(TRAIN + "demographics.csv").set_index("patient_id")
    rm = pd.read_csv(TRAIN + "rnaseq_metadata.csv", parse_dates=["collection_date"])
    re_ = pd.read_csv(TRAIN + "rnaseq_embeddings.csv").merge(rm[["rnaseq_sample_id", "patient_id", "collection_date"]], on="rnaseq_sample_id")
    emb = [c for c in re_.columns if c.startswith("rnaseq_dim_")]
    b = t[t.treatment_class == "Biologic"].merge(re_[["patient_id", "collection_date"] + emb], on="patient_id")
    b = b[b.start_date == b.collection_date].drop_duplicates("patient_id")
    sc = {p: g for p, g in v.groupby("patient_id")}
    rows, courses = [], []
    for r in b.itertuples():
        g = sc.get(r.patient_id)
        if g is None:
            continue
        d = (g.visit_date - r.start_date).dt.days
        pre, post = g.ihs4_score[(d >= -180) & (d < 0)], g.ihs4_score[(d >= 90) & (d <= 270)]
        if len(pre) == 0 or len(post) == 0 or pre.mean() < 3:
            continue
        prior = g.ihs4_score[d < 0]
        dm = demo.loc[r.patient_id]
        rows.append(dict(ratio=post.mean() / pre.mean(), B=pre.mean(), n_pre=len(pre), last_pre=prior.iloc[-1], mean_all_pre=prior.mean(),
                         age=r.start_date.year - dm.birth_year, sex=float(dm.sex == "Female"), bmi=dm.bmi_at_baseline,
                         hurley=dm.hurley_stage_baseline, delay=dm.diagnostic_delay_years, line=r.treatment_line, drug=r.drug_name,
                         **{c: getattr(r, c) for c in emb}))
        courses.append((d.to_numpy(), g.ihs4_score.to_numpy() / pre.mean()))
    return pd.DataFrame(rows), courses, re_[emb].to_numpy(float), emb


def smd(Z, l):
    return (Z[l].mean(0) - Z[~l].mean(0)) / np.sqrt((Z[l].var(0) * l.sum() + Z[~l].var(0) * (~l).sum()) / len(l))


def oof_auc(C, Z, lab, top=False, reps=5):
    """out-of-fold AUC of a logistic model on C (+ the component of Z with the largest |SMD| in the training fold if top)"""
    aucs = []
    for rep in range(reps):
        P = np.zeros(len(lab))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(C):
            A = np.column_stack([C, Z[:, np.argmax(np.abs(smd(Z[tr], lab[tr])))]]) if top else C
            P[te] = LogisticRegression(C=1.0, max_iter=500).fit(A[tr], lab[tr]).predict_proba(A[te])[:, 1]
        aucs.append(roc_auc_score(lab, P))
    return float(np.mean(aucs))


def course_median(courses, idx, edges, rng, min_n=10, boot=1000):
    """per bin of days from the start: the median over the starts idx of their mean relative score in the bin (bins with >= min_n starts), with a
    bootstrap 95% interval over starts; also the starts per bin and the bins kept"""
    M = np.full((len(idx), len(edges) - 1), np.nan)
    for a, i in enumerate(idx):
        dd, rr = courses[i]
        for j in range(len(edges) - 1):
            w = (dd >= edges[j]) & (dd < edges[j + 1])
            if w.any():
                M[a, j] = rr[w].mean()
    cnt = np.isfinite(M).sum(0)
    ok = cnt >= min_n
    med, lo, hi = (np.full(len(cnt), np.nan) for _ in range(3))
    med[ok] = np.nanmedian(M[:, ok], 0)
    bs = np.array([np.nanmedian(M[rng.integers(0, len(idx), len(idx))][:, ok], 0) for _ in range(boot)])
    lo[ok], hi[ok] = np.nanpercentile(bs, [2.5, 97.5], 0)
    return med, lo, hi, cnt, ok


def fig_column(courses, third, rng, out):
    """The third plot of the three-column slide (dark, non-technical; sde_plots.py --figs K makes the other two): the median course relative to
    the 6 months before the start for the lowest (profile A) and the highest (profile B) third of PC1, 3-month bins with >= 20 starts."""
    D = {"bg": "#111827", "ax": "#C9D1DC", "grid": "#253041", "spine": "#3A4556", "a": "#3987E5", "b": "#C9D1DC"}
    plt.rcParams.update({"font.size": 16})
    edges = np.arange(-6, 10, 3) * 30.44
    centers = (edges[:-1] + edges[1:]) / 2 / 30.44
    fig, ax = plt.subplots(figsize=(5.0, 3.9), facecolor=D["bg"])
    ax.set_facecolor(D["bg"])
    for k, name, col, word in ((0, "RNAseq profile A", D["a"], "halved"), (2, "RNAseq profile B", D["b"], "little change")):
        med, lo, hi, cnt, ok = course_median(courses, np.flatnonzero(third == k), edges, rng, min_n=20)
        ax.fill_between(centers[ok], 100 * lo[ok], 100 * hi[ok], color=col, alpha=0.18, lw=0)
        ax.plot(centers[ok], 100 * med[ok], "-o", color=col, lw=3, ms=7, label=name)
        last = np.flatnonzero(ok)[-1]
        ax.text(centers[last] + 0.35, 100 * med[last], word, color=col, fontsize=15, va="center")
        print(f"column: {name}: starts per bin {cnt.tolist()}; median {np.round(med, 2).tolist()}")
    ax.axvline(0, color="#8A94A6", lw=1.2, ls=":")
    ax.set_xlim(-6.3, 10.8)
    ax.set_ylim(20, 150)
    ax.set_xticks([-6, 0, 6])
    ax.set_xticklabels(["6 months\nbefore", "first dose", "6 months\nafter"], fontsize=14)
    ax.set_yticks([25, 50, 75, 100, 125])
    ax.set_yticklabels(["25%", "50%", "75%", "100%", "125%"])
    ax.set_ylabel("disease score, % of before", color=D["ax"], fontsize=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(D["spine"])
    ax.tick_params(colors=D["ax"])
    ax.grid(axis="y", color=D["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=13, labelcolor=D["ax"], ncol=1, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=D["bg"])
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def style(ax):
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/rna_pc1")
    ap.add_argument("--columns", default=None, help="also write the dark column plot of profile A / B to this file")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    D, courses, E_all, emb = starts()
    pca = PCA(20).fit(E_all)
    Z = pca.transform(D[emb].to_numpy(float)) / np.sqrt(pca.explained_variance_[None, :])
    pc1, y = Z[:, 0], D.ratio.to_numpy()
    lab, drug = y < 0.5, D.drug.to_numpy()
    third = np.digitize(pc1, np.quantile(pc1, [1 / 3, 2 / 3]))  # 0 low, 1 middle, 2 high
    cols = np.array([C_LOW, C_MID, C_HIGH])
    names = ["low PC1", "middle PC1", "high PC1"]

    # statistics
    rho = [spearmanr(Z[:, j], y)[0] for j in range(20)]
    null1, nullmax = np.zeros(10000), np.zeros(10000)
    for b in range(10000):
        ys = rng.permutation(y)
        r = np.array([spearmanr(Z[:, j], ys)[0] for j in range(20)]) if b < 2000 else None
        null1[b] = spearmanr(pc1, ys)[0]
        if r is not None:
            nullmax[b] = np.abs(r).max()
    p1 = (1 + np.sum(np.abs(null1) >= abs(rho[0]))) / (1 + len(null1))
    pmax = (1 + np.sum(nullmax[:2000] >= abs(rho[0]))) / 2001
    clin = ["B", "n_pre", "last_pre", "mean_all_pre", "age", "sex", "bmi", "hurley", "delay", "line"]
    C = D[clin].astype(float)
    C = ((C.fillna(C.median()) - C.mean()) / C.std()).to_numpy()
    auc_c, auc_cz = oof_auc(C, Z, lab), oof_auc(C, Z, lab, top=True)
    lines = [f"{len(D)} starts, {lab.sum()} good responders; " + ", ".join(f"{k} {int((drug == k).sum())}" for k in MARK),
             f"Spearman(PC1, ratio) = {rho[0]:+.3f}; p (10,000 shuffles) = {p1:.4f}; p against the largest |rho| of 20 components "
             f"(2,000 shuffles) = {pmax:.4f}",
             "Spearman of PC2..PC5: " + ", ".join(f"PC{j + 1} {rho[j]:+.3f}" for j in range(1, 5)),
             "per drug: " + ", ".join(f"{k} rho {spearmanr(pc1[drug == k], y[drug == k])[0]:+.2f} (n={int((drug == k).sum())})" for k in MARK),
             f"out-of-fold AUC (good responder): clinical {auc_c:.3f}; clinical + top component (selected in each training fold) {auc_cz:.3f}"]
    for k in range(3):
        s = third == k
        p, lo, hi = wilson(lab[s].sum(), s.sum())
        lines.append(f"{names[k]:10s} n={s.sum():3d}  good share {p:.2f} [{lo:.2f}, {hi:.2f}]  median ratio {np.median(y[s]):.2f}  "
                     f"mean pre-start score {D.B[s].mean():.1f}")
    print("\n".join(lines))
    open(f"{args.out}/report.txt", "w").write("\n".join(lines) + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    # A. ratio against PC1
    ax = axs[0]
    style(ax)
    ytop = 2.0
    for k, m in MARK.items():
        s = drug == k
        ax.scatter(pc1[s], np.minimum(y[s], ytop), c=cols[third[s]], marker=m, s=34, alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter([], [], marker=m, s=34, color="#666666", label=k)
    q = np.quantile(pc1, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(pc1, q[1:-1]), 0, 4)
    cx = np.array([np.median(pc1[bins == j]) for j in range(5)])
    med = np.array([np.median(y[bins == j]) for j in range(5)])
    bs = np.array([[np.median(rng.choice(y[bins == j], (bins == j).sum())) for j in range(5)] for _ in range(2000)])
    ax.fill_between(cx, *np.percentile(bs, [2.5, 97.5], 0), color=C_INK, alpha=0.10, lw=0, zorder=2)
    ax.plot(cx, med, "-o", color=C_INK, lw=2, ms=6, zorder=4, label="median per fifth of PC1 (95% interval)")
    ax.axhline(0.5, color="#666666", lw=1, ls="--", zorder=1, label="good responder: score halved")
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
    ax.set_ylim(0, ytop + 0.05)
    ax.set_xlabel("RNAseq PC1 at the start (standard deviations)")
    ax.set_ylabel(f"response ratio: score months 3-9 / 6 months before\n(capped at {ytop:.0f})")
    ax.set_title(f"A. Lower PC1, larger drop of the score\nSpearman ρ = {rho[0]:+.2f}, p = {p1:.4f} ({len(D)} starts)", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    # B. courses per third
    ax = axs[1]
    style(ax)
    edges = np.arange(-6, 13, 3) * 30.44
    centers = (edges[:-1] + edges[1:]) / 2 / 30.44
    ax.axvspan(90 / 30.44, 270 / 30.44, color="#f2f2f2", zorder=0)
    ax.text(180 / 30.44, 1.27, "response window", ha="center", fontsize=9, color="#555555")
    ax.axvline(0, color="#999999", lw=1, ls=":")
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    for k in (1, 0, 2):
        idx = np.flatnonzero(third == k)
        med, lo, hi, cnt, ok = course_median(courses, idx, edges, rng)
        print(f"B. {names[k]}: patients per bin {cnt.tolist()}; median {np.round(med, 2).tolist()}")
        if k != 1:
            ax.fill_between(centers[ok], lo[ok], hi[ok], color=cols[k], alpha=0.15, lw=0)
        ax.plot(centers[ok], med[ok], "-o", color=cols[k], lw=2 if k != 1 else 1.4, ms=5 if k != 1 else 3.5, zorder=3)
        last = np.flatnonzero(ok)[-1]
        nudge = {0: 0.0, 1: -0.035, 2: 0.035}[k]  # the middle and high lines end close together
        ax.text(centers[last] + 0.4, med[last] + nudge, f"{names[k]}\n(n={len(idx)})", color=cols[k] if k != 1 else "#666666", fontsize=9,
                va="center")
    ax.set_xlim(-6, 13.5)
    ax.set_ylim(0.3, 1.32)
    ax.set_xticks(np.arange(-6, 13, 3))
    ax.set_xlabel("months from the biologic start")
    ax.set_ylabel("IHS4 / mean of the 6 months before")
    ax.set_title("B. The courses separate after the first dose\nthirds of PC1, median per 3 months with 95% band", loc="left", fontsize=11)
    # C. good responder share per third
    ax = axs[2]
    style(ax)
    for k in range(3):
        s = third == k
        p, lo, hi = wilson(lab[s].sum(), s.sum())
        ax.bar(k, p, width=0.62, color=cols[k], zorder=2)
        ax.errorbar(k, p, yerr=[[p - lo], [hi - p]], color=C_INK, capsize=4, lw=1.2, zorder=3)
        ax.text(k, hi + 0.015, f"{100 * p:.0f}%\n{lab[s].sum()}/{s.sum()}", ha="center", va="bottom", fontsize=9.5, color=C_INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("share of good responders (95% interval)")
    ax.set_title(f"C. Good responders per third\nout-of-fold AUC: clinical {auc_c:.2f}, + RNAseq {auc_cz:.2f}", loc="left", fontsize=11)
    per = ", ".join(f"{k} ρ = {spearmanr(pc1[drug == k], y[drug == k])[0]:+.2f} (n={int((drug == k).sum())})" for k in MARK)
    fig.text(0.01, 0.01, f"Biologic starts with a baseline RNAseq sample (train folder). PC1 of the 128-d RNAseq embedding (PCA on all 600 samples, "
             f"no labels); PC2 to PC5: |ρ| < 0.07. Per drug: {per}. Association at baseline; no untreated patients have RNAseq.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{args.out}/PC1_response.png", dpi=150)
    print(f"saved {args.out}/PC1_response.png")
    if args.columns:
        fig_column(courses, third, rng, args.columns)
        print(f"saved {args.columns}")


if __name__ == "__main__":
    main()
