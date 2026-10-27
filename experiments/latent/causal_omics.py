"""
Exploratory: do the RNAseq or image components change the effect of Bimekizumab against Secukinumab? (default env, CPU, < 1 min)
Small samples (line 2: about 70 starts with each modality), so the tests use the near-random choice of the drug: p-values come from shuffling the drug
label inside the subset (2,000 shuffles), which needs no model assumption.

Unit: line 2 Bimekizumab / Secukinumab starts with the modality (RNAseq: sample on the start day; image: the latest image <= 365 days before the start).
Outcome: the ratio post / pre (capped at 3; lower = better). O2 = days 30..270 after the start, pre mean >= 1 (main here: more starts); O1 = days 90..270,
pre mean >= 3. Components: the first 10 principal components of each embedding (fitted on all train samples), standardized in the subset.
Per component k: ratio ~ 1 + drug + z_k + drug x z_k + pre mean (OLS); the drug x z_k coefficient = the change of the effect (Bime - Secu, ratio) per SD.
  - RNAseq PC1 is tested alone (pre-specified: the Adalimumab response component of rna_profiles.py, oriented so that higher = worse Adalimumab
    response, as there);
  - all 10 components: the max |t| over components against its shuffle distribution (one global p, and per-component adjusted p);
  - a leave-one-out drug-choice rule (ridge per drug on 5 components): AIPW value of "give the predicted better drug" minus a coin flip, shuffle p.
Sensitivity: lines 2 and 3 together (at line 3 the drug is the one not used before, so a drug x component term can mix with the prior drug).
Writes runs/causal_omics/*.png (scatter of the ratio against a component, per drug).

    python -I experiments/latent/causal_omics.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
OUT = "runs/causal_omics"
os.makedirs(OUT, exist_ok=True)
d = dict(np.load("test_data/hs_public_dataset/hs_e2e_train.npz"))
names = [str(n) for n in d["feature_names"]]
X0, M, kind, day, off = d["X"].astype(np.float64), d["M"] > 0, d["kind"], d["day"].astype(np.int64), d["offsets"]
prow = np.repeat(np.arange(len(off) - 1), np.diff(off))
own = names.index("own_score")
has = (kind == 0) & M[:, own]
key = {(int(p), int(s)): (int(k), int(l)) for p, s, k, l in zip(d["bio_patient"], d["bio_start"].astype(np.int64), d["bio_drug"], d["bio_line"])}
K = 10
rp, ip = PCA(K).fit(d["rna_emb"]), PCA(K).fit(d["img_emb"])
j_img = names.index("img_days_since")


def outcome(r, lo, hi, minpre):
    i, s = prow[r], day[r] + 1
    sel = np.arange(off[i], off[i + 1])[has[off[i]:off[i + 1]]]
    dd, sv = day[sel] - s, X0[sel, own]
    pre, post = sv[(dd >= -180) & (dd < 0)], sv[(dd >= lo) & (dd <= hi)]
    return (min(post.mean() / pre.mean(), 3.0), pre.mean()) if len(pre) and len(post) and pre.mean() >= minpre else (np.nan, np.nan)


def collect(lines, lo, hi, minpre, drugs=(1, 2)):
    out = {"rna": [], "img": []}
    for r in np.flatnonzero(kind == 3):
        k = key.get((int(prow[r]), int(day[r]) + 1))
        if k is None or k[0] not in drugs or k[1] not in lines:
            continue
        y, pre = outcome(r, lo, hi, minpre)
        if not np.isfinite(y):
            continue
        if d["rna_row"][r] >= 0:
            out["rna"].append((int(k[0] == 1), y, pre, k[1], rp.transform(d["rna_emb"][[d["rna_row"][r]]])[0], int(k[0])))
        if d["img_row"][r] >= 0 and M[r, j_img] and X0[r, j_img] <= 365:
            out["img"].append((int(k[0] == 1), y, pre, k[1], ip.transform(d["img_emb"][[d["img_row"][r]]])[0], int(k[0])))
    return out


# orientation of RNAseq PC1 as in rna_profiles.py: higher = worse Adalimumab response (fitted on the Adalimumab starts, which are not used below)
ada = collect((1,), 30, 270, 1, drugs=(0,))["rna"]
s1 = np.sign(np.corrcoef([x[4][0] for x in ada], [x[1] for x in ada])[0, 1])
print(f"RNAseq PC1 orientation from {len(ada)} Adalimumab starts: corr(PC1, ratio) sign {s1:+.0f} (PC1 is multiplied by it)")


def tstats(A, y, pre, Z, line=None):
    """t of the drug x z_k term for every column k (HC1)"""
    n = len(y)
    out = np.zeros(Z.shape[1])
    for k in range(Z.shape[1]):
        cols = [np.ones(n), A, Z[:, k], A * Z[:, k], (pre - pre.mean()) / pre.std()]
        if line is not None and np.std(line) > 0:
            cols.append(line - line.mean())
        D = np.column_stack(cols)
        G = np.linalg.pinv(D.T @ D)
        b = G @ D.T @ y
        e = y - D @ b
        V = G @ (D.T * e ** 2) @ D @ G * n / max(n - D.shape[1], 1)
        out[k] = b[3] / np.sqrt(V[3, 3])
    return out


def coef(A, y, pre, z):
    D = np.column_stack([np.ones(len(y)), A, z, A * z, (pre - pre.mean()) / pre.std()])
    return (np.linalg.pinv(D.T @ D) @ D.T @ y)[3]


def ridge_pred(Xtr, ytr, Xte, lam=5.0):
    Xm, ym = Xtr.mean(0), ytr.mean()
    Xc = Xtr - Xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (ytr - ym))
    return ym + (Xte - Xm) @ w


def loo_policy(A, y, Z):
    """leave-one-out tau from ridge per drug on 5 components; AIPW value of 'give the predicted better drug' minus a coin flip (outcome: minus ratio)"""
    n, yb = len(y), -y
    m1, m0 = np.zeros(n), np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        for a, m in ((1, m1), (0, m0)):
            t2 = tr & (A == a)
            m[i] = ridge_pred(Z[t2, :5], yb[t2], Z[i:i + 1, :5])[0]
    e = A.mean()
    pol = (m1 > m0).astype(float)
    b = m1 + A * (yb - m1) / e
    s = m0 + (1 - A) * (yb - m0) / (1 - e)
    return (pol * b + (1 - pol) * s).mean() - (0.5 * b + 0.5 * s).mean()


def run(title, rows, modality, line_cov=False, n_perm=2000, plot=True):
    if len(rows) < 20:
        print(f"\n{title}: {len(rows)} starts, too few")
        return
    A = np.array([x[0] for x in rows]); y = np.array([x[1] for x in rows]); pre = np.array([x[2] for x in rows])
    line = np.array([x[3] for x in rows], float) if line_cov else None
    Z = np.array([x[4] for x in rows])
    if modality == "rna":
        Z[:, 0] *= s1
    Z = (Z - Z.mean(0)) / Z.std(0)
    t = tstats(A, y, pre, Z, line)
    null = np.array([np.abs(tstats(rng.permutation(A), y, pre, Z, line)).max() for _ in range(n_perm)])
    t1_null = None
    print(f"\n{title}: {len(y)} starts (Bimekizumab {A.sum()}, Secukinumab {(1 - A).sum()}); mean ratio Bime {y[A == 1].mean():.3f}, Secu {y[A == 0].mean():.3f}")
    print(f"   {'component':10s} {'drug x comp. (ratio per SD)':>28s} {'t':>6s} {'adjusted p (max |t|)':>21s}")
    for k in range(K):
        print(f"   PC{k + 1:<8d} {coef(A, y, pre, Z[:, k]):+28.3f} {t[k]:+6.2f} {np.mean(null >= abs(t[k])):21.3f}")
    print(f"   global test (any component): max |t| = {np.abs(t).max():.2f}, shuffle p = {np.mean(null >= np.abs(t).max()):.3f}")
    if modality == "rna":
        t1_null = np.array([tstats(rng.permutation(A), y, pre, Z[:, :1], line)[0] for _ in range(n_perm)])
        print(f"   pre-specified RNAseq PC1 alone: t = {t[0]:+.2f}, shuffle p (two-sided) = {np.mean(np.abs(t1_null) >= abs(t[0])):.3f}")
        for a, nm in ((1, "Bimekizumab"), (0, "Secukinumab")):
            s = A == a
            print(f"      slope of the ratio on PC1 inside {nm:12s}: {np.polyfit(Z[s, 0], y[s], 1)[0]:+.3f} per SD (n={s.sum()})")
    v = loo_policy(A, y, Z)
    vn = np.array([loo_policy(rng.permutation(A), y, Z) for _ in range(300)])
    print(f"   leave-one-out drug-choice rule (5 components): value minus coin flip {v:+.3f} (minus ratio), shuffle p = {np.mean(vn >= v):.3f}")
    if plot:
        pre_k, am = (0 if modality == "rna" else 1), int(np.argmax(np.abs(t)))  # the pre-specified component, the largest |t|
        ks = [pre_k] + ([am] if am != pre_k else [])
        fig, axes = plt.subplots(1, len(ks), figsize=(5.6 * len(ks), 4.4), squeeze=False)
        for ax, k in zip(axes[0], ks):
            for a, nm, c in ((1, "Bimekizumab", "#2b6cb0"), (0, "Secukinumab", "#c05621")):
                s = A == a
                ax.scatter(Z[s, k], y[s], s=26, color=c, alpha=0.8, edgecolor="white", lw=0.6, label=f"{nm} (n={s.sum()})")
                xx = np.linspace(Z[:, k].min(), Z[:, k].max(), 20)
                ax.plot(xx, np.polyval(np.polyfit(Z[s, k], y[s], 1), xx), color=c, lw=2)
            ax.axhline(0.5, color="#999999", lw=1, ls=":")
            why = "pre-specified" if (modality == "rna" and k == 0) or (modality == "img" and k == 1) else "largest |t| (chosen after the fit)"
            ax.set_title(f"{'RNAseq' if modality == 'rna' else 'image'} PC{k + 1} ({why}): t = {t[k]:+.2f}, adj. p = {np.mean(null >= abs(t[k])):.2f}",
                         loc="left", fontsize=9)
            ax.set_xlabel(f"PC{k + 1} (standardized)")
            ax.set_ylabel("ratio post / pre (lower = better)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fn = f"{OUT}/{modality}_{title.split(':')[0].replace(' ', '_')}.png"
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        print(f"   plot: {fn}")


for tag, (lo, hi, mp) in (("O2", (30, 270, 1)), ("O1", (90, 270, 3))):
    c2 = collect((2,), lo, hi, mp)
    run(f"{tag} line2 RNAseq: days {lo}..{hi}, pre >= {mp}", c2["rna"], "rna", plot=(tag == "O2"))
    run(f"{tag} line2 image: days {lo}..{hi}, pre >= {mp}", c2["img"], "img", plot=(tag == "O2"))
c23 = collect((2, 3), 30, 270, 1)
run("O2 lines2-3 RNAseq (sensitivity, line as covariate)", c23["rna"], "rna", line_cov=True, plot=False)
run("O2 lines2-3 image (sensitivity, line as covariate)", c23["img"], "img", line_cov=True, plot=False)
