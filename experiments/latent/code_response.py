"""
Is the patient trait visible before a biologic start? (patient-code model; no training; one short GPU job, ~1-2 min)

Held-out patients only (never used in training): a random 900 of their biologic starts (hs_groups.find_starts; ratio = mean score days
90..270 after the start / mean 180 days before, pre mean >= 3; good responder: ratio < 0.5). Per start:
  start features   hs_task2.PROFILE + drug at the decision step (the day before the start)
  m                the posterior mean at the decision step (16 PCs; a forward filter: no later data)
  c before         the patient code fitted on the data up to the decision step only (fit_codes with cuts, 30 Adam steps, lr 0.15:
                   codes correlate 0.93 with 100 steps at lr 0.05)
1. Out-of-fold AUC (logistic, 5 folds grouped by patient) with a bootstrap over patients; paired differences against the start features.
2. What c before knows: cross-validated R2 (ridge, 5 folds) from c before to features at the decision step.
3. RNAseq in the trained codes (training patients with RNAseq, saved codes, CPU): cross-validated R2 from c to the RNAseq PCs.

    python -m experiments.latent.code_response [runs/hs_fused_sde_<stamp>]
"""
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.latent.hs_e2e import PATHS, Data
from experiments.latent.hs_fused_sde import fit_codes, make_model, prepare
from experiments.latent.hs_groups import find_starts
from experiments.latent.hs_task2 import PROFILE


def oof(X, y, g):
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), X, y, cv=GroupKFold(5), groups=g,
                             method="predict_proba")[:, 1]


def boot(fn, g, n=500, seed=0):
    rng = np.random.default_rng(seed)
    up = np.unique(g)
    ix = {p: np.flatnonzero(g == p) for p in up}
    return np.percentile([fn(np.concatenate([ix[p] for p in rng.choice(up, len(up))])) for _ in range(n)], [2.5, 97.5])


def r2cv(X, f):
    ok = np.isfinite(f)
    pr = cross_val_predict(RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)), StandardScaler().fit_transform(X[ok]), f[ok],
                           cv=KFold(5, shuffle=True, random_state=0))
    return 1 - ((f[ok] - pr) ** 2).sum() / ((f[ok] - f[ok].mean()) ** 2).sum(), ok.sum()


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_fused_sde_20261026-234554"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{run}/model.pt", map_location=dev, weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    nm = data.names.index

    # 3. RNAseq in the trained codes (saved, no fit)
    pool = np.random.default_rng(ck["args"].get("seed", 0)).permutation(np.flatnonzero(~data.flag))
    tr = np.sort(pool[int(0.1 * len(pool)):])
    C = model.pcode.weight.detach().cpu().numpy()
    E = data.rna_emb.astype(np.float64)
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    pcs = U[:, :3] / U[:, :3].std(0)
    rp = np.array([i for i in tr if data.rna_row[data.final_row(i)] >= 0])
    rr = data.rna_row[[data.final_row(i) for i in rp]]
    lvl = np.array([np.nanmean(data.score[data.off[i]:data.off[i + 1]][data.has_score[data.off[i]:data.off[i + 1]]]) for i in rp])
    keep = np.isfinite(lvl)  # patients without a scored visit have no mean score
    rp, rr, lvl = rp[keep], rr[keep], lvl[keep]
    print(f"model {run}\n\n3. RNAseq in the trained codes ({len(rp)} training patients with RNAseq; codes fitted on the whole record)")
    for k in range(3):
        a, _ = r2cv(C[rp], pcs[rr, k])
        b, _ = r2cv(lvl[:, None], pcs[rr, k])
        print(f"   RNAseq PC{k + 1}: R2 from c {a:+.3f}   (from the mean visit score alone {b:+.3f})")

    te = np.flatnonzero(data.flag)
    st = find_starts(data, te)
    st = [st[k] for k in np.sort(np.random.default_rng(0).choice(len(st), min(900, len(st)), replace=False))]
    pat = np.array([s[0] for s in st])
    kd = np.array([s[3] for s in st])
    good = np.array([s[4] < 0.5 for s in st])
    drug = np.array([s[2] for s in st])
    rows = np.array([data.off[p] + k for p, k in zip(pat, kd)])
    B = np.column_stack([np.where(data.M[rows, nm(c)], data.raw[rows, nm(c)], np.nan) for c in PROFILE] +
                        [(drug == d).astype(float) for d in np.unique(drug)])
    B = np.where(np.isfinite(B), B, np.nanmedian(B, 0))
    M = np.zeros((len(st), model.dz), np.float32)
    up = np.unique(pat)
    with torch.no_grad():
        for a in range(0, len(up), 50):
            sel = [int(i) for i in up[a:a + 50]]
            model.encode(data.pack(sel, dev))
            m = model.last_post[0].cpu().numpy()
            for b, i in enumerate(sel):
                for j in np.flatnonzero(pat == i):
                    M[j] = m[b, kd[j]]
    Mz = StandardScaler().fit_transform(M)
    Um, Sm, _ = np.linalg.svd(Mz - Mz.mean(0), full_matrices=False)
    M16 = Um[:, :16] * Sm[:16]
    t0 = time.time()
    Cpre = fit_codes(model, data, pat, dev, steps=30, lr=0.15, cuts=kd)
    print(f"\n{len(st)} biologic starts of {len(up)} held-out patients (good responders {good.mean():.3f}); codes fitted on the data before "
          f"each start in {time.time() - t0:.0f}s on {dev}; |c| {np.linalg.norm(Cpre, axis=1).mean():.2f}")

    print("\n1. Response: out-of-fold AUC [95% bootstrap over patients]; paired difference against the start features")
    P = {"c before the start (alone)": oof(Cpre, good, pat), "start features": oof(B, good, pat), "m at the decision step": oof(M16, good, pat),
         "start features + c before": oof(np.column_stack([B, Cpre]), good, pat),
         "start features + m": oof(np.column_stack([B, M16]), good, pat),
         "start features + m + c before": oof(np.column_stack([B, M16, Cpre]), good, pat)}
    base = P["start features"]
    for k, p in P.items():
        lo, hi = boot(lambda ix: roc_auc_score(good[ix], p[ix]), pat)
        line = f"   {k:32s} AUC {roc_auc_score(good, p):.3f} [{lo:.3f}, {hi:.3f}]"
        if k.startswith("start features +"):
            dl, dh = boot(lambda ix: roc_auc_score(good[ix], p[ix]) - roc_auc_score(good[ix], base[ix]), pat)
            line += f"   gain {roc_auc_score(good, p) - roc_auc_score(good, base):+.3f} [{dl:+.3f}, {dh:+.3f}]"
        print(line)

    print("\n2. What c before the start knows: cross-validated R2 (ridge, 5 folds) from c before to features at the decision step")
    for c, name in (("mean_6m", "recent level (6-month mean)"), ("mean_all", "long-term level (mean of all scores)"),
                    ("std_6m", "spread of the scores (6 months)"), ("vis6m_flare_reactive", "flare visits (6 months)"),
                    ("n_surg", "surgeries so far"), ("hurley_stage_baseline", "Hurley stage")):
        a, n = r2cv(Cpre, np.where(data.M[rows, nm(c)], data.raw[rows, nm(c)], np.nan))
        print(f"   {name:38s} R2 {a:+.3f}  (n={n})")


if __name__ == "__main__":
    main()
