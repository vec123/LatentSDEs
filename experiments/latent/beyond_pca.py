"""
Is there outcome information in the embedding directions that PCA drops? (no training; CPU < 1 min)

RNAseq: the 600 train-folder patients with a sample before the cutoff; target = real label minus the forecast of a saved fused model with the
RNAseq hidden (what the RNAseq must explain). Image: train-folder patients at the first scored visit with a new image; target = true visit
score minus the model's score prediction from the history (the visit score and the image hidden at that step). Cross-validated ridge (5 folds,
3 repeats, alpha by inner grid) from blocks of PCA directions of the standardized embedding (PCA on all samples, no labels): the first 2, the
first 16, all except the first 16, all. If the dropped block predicts nothing, a learned compression has nothing to find beyond the top PCs.

    python -m experiments.latent.beyond_pca [runs/hs_fused_sde_<stamp>]
"""
import sys

import numpy as np
import torch

from experiments.latent.hs_e2e import PATHS, Data
from experiments.latent.hs_fused_sde import make_model, prepare
from experiments.latent.hs_discovery import forecast


def cv_r2(X, y, reps=3, alphas=(1.0, 10.0, 100.0, 1000.0)):
    out = []
    for r in range(reps):
        folds = np.random.default_rng(r).integers(0, 5, len(y))
        best = -np.inf
        for a in alphas:
            P = np.zeros(len(y))
            for f in range(5):
                tr, te = folds != f, folds == f
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                A = (X[tr] - mu) / sd
                W = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
                P[te] = ((X[te] - mu) / sd) @ W + y[tr].mean()
            best = max(best, 1 - ((y - P) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out.append(best)
    return np.mean(out)


def blocks(E):
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    Z = U * S
    return {"PC 1-2": Z[:, :2], "PC 1-16": Z[:, :16], "PC 17+ (dropped by PCA)": Z[:, 16:], "all directions": Z}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/hs_fused_sde_20261026-210827"
    torch.set_num_threads(4)
    ck = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"])
    model.load_state_dict(ck["model"])
    model.eval()

    pats = np.array([i for i in range(data.n_pat) if data.rna_row[data.final_row(i)] >= 0])
    fr = np.array([data.final_row(i) for i in pats])
    res = data.y[fr] - forecast(model, data, pats, "cpu", hide_rna=True)
    B = blocks(data.rna_emb.astype(np.float64))
    print(f"model {run}\nRNAseq (n={len(pats)}): CV R2 of (real label - forecast without RNAseq) from blocks of RNAseq PCA directions")
    for k, Z in B.items():
        print(f"   {k:26s} ({Z.shape[1]:3d} dims)  R2 {cv_r2(Z[data.rna_row[fr]], res):+.3f}")

    own = data.names.index("own_score")
    rows, ps = [], []
    for i in range(data.n_pat):
        s = slice(data.off[i], data.off[i + 1])
        r = np.flatnonzero(data.img_new[s] & data.has_score[s])
        if len(r):
            rows.append(data.off[i] + r[0])
            ps.append(i)
    rows, ps = np.array(rows), np.array(ps)
    X0, M0 = data.X[rows, own].copy(), data.Mf[rows, own].copy()
    data.X[rows, own], data.Mf[rows, own] = 0, 0
    P = np.zeros(len(rows))
    with torch.no_grad():
        for a in range(0, len(ps), 200):
            sel = [int(i) for i in ps[a:a + 200]]
            model.encode(data.pack(sel, "cpu", hide_img=True))
            m = model.last_post[0]
            k = torch.as_tensor([rows[a + b] - data.off[i] for b, i in enumerate(sel)])
            mu = model.decode(m[torch.arange(len(sel)), k], torch.as_tensor(data.vt[rows[a:a + len(sel)]]))[0].numpy()
            P[a:a + len(sel)] = np.expm1(mu * data.stats["ls"] + data.stats["lm"])
    data.X[rows, own], data.Mf[rows, own] = X0, M0
    res = data.score[rows] - P
    B = blocks(data.img_emb.astype(np.float64))
    print(f"\nImage (n={len(rows)}, first scored visit with a new image): CV R2 of (visit score - model prediction from the history) from "
          f"blocks of image PCA directions")
    for k, Z in B.items():
        print(f"   {k:26s} ({Z.shape[1]:3d} dims)  R2 {cv_r2(Z[data.img_row[rows]], res):+.3f}")


if __name__ == "__main__":
    main()
