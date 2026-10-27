"""
How much RNAseq information does each model keep in the latent that its forecast head reads? (no training; CPU, ~2 min)

Patients: the train-folder patients with an RNAseq sample before the cutoff. Latent at the final step: the GRU state (hs_e2e.py) or the
posterior mean m_t (hs_fused_sde.py), once with the RNAseq shown and once hidden. Target: the first 5 principal components of the RNAseq
embedding (PCA on all samples, no labels; PC1 is the response axis of section 16). Ridge probe, 5-fold cross-validation by patient: R2 shown,
R2 hidden (what the EHR and the history already tell), and the difference (the RNAseq information the latent keeps).
Also the head: mean |forecast shown - forecast hidden| (score points), and on the held-out patients the gain (MAE hidden - MAE shown).

    python -m experiments.latent.rna_use_check
"""
import numpy as np
import torch

from experiments.latent import hs_e2e, hs_fused_sde
from experiments.latent.hs_e2e import PATHS, Data

RUNS = [("GRU raw, 16 PCs", "gru", "runs/hs_e2e_20261026-174232"), ("GRU raw, full", "gru", "runs/hs_e2e_20261026-181520"),
        ("GRU raw, full + recon", "gru", "runs/hs_e2e_20261026-185512"),
        ("Fused A: no SDE terms (features)", "fused", "runs/hs_fused_sde_20261026-172329"),
        ("Fused B cont.: SDE, beta 1 (features)", "fused", "runs/hs_fused_sde_20261026-180747"),
        ("Fused a: SDE, beta 1 (raw)", "fused", "runs/hs_fused_sde_20261026-175319"),
        ("Fused + recon: SDE, beta 1 (raw)", "fused", "runs/hs_fused_sde_20261026-190817"),
        ("Fused + head skip: SDE, beta 1 (raw)", "fused", "runs/hs_fused_sde_20261026-210827")]


def ridge_cv_r2(X, Y, folds, alphas=(0.1, 1.0, 10.0, 100.0)):
    """cross-validated R2 per target column, the best alpha of the grid (the same grid for every model)"""
    best = None
    for a in alphas:
        P = np.zeros_like(Y)
        for k in range(folds.max() + 1):
            tr, te = folds != k, folds == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            A = (X[tr] - mu) / sd
            ym = Y[tr].mean(0)
            W = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
            P[te] = ((X[te] - mu) / sd) @ W + ym
        r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        best = r2 if best is None or r2.mean() > best.mean() else best
    return best


def latents(kind, model, data, pats, hide_rna):
    """latent and head forecast (score points) at the final step of each patient"""
    Z, F = [], []
    with torch.no_grad():
        for a in range(0, len(pats), 50):
            sel = [int(i) for i in pats[a:a + 50]]
            b = data.pack(sel, "cpu", hide_rna=hide_rna)
            h = model.encode(b)
            z = h if kind == "gru" else model.last_post[0]
            mu = model.forecast(h)[0]
            for j, i in enumerate(sel):
                k = data.n_rows(i) - 1
                Z.append(z[j, k].numpy())
                F.append(float(mu[j, k]) * data.stats["ystd"] + data.stats["ymean"])
    return np.array(Z), np.array(F)


def main():
    torch.set_num_threads(2)
    raw = np.load(PATHS["train"])
    E = raw["rna_emb"].astype(np.float64)
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    U, S, _ = np.linalg.svd(E, full_matrices=False)
    pcs = U[:, :5] * S[:5]
    pcs = (pcs - pcs.mean(0)) / pcs.std(0)
    print(f"{'model':40s} {'R2 PC1 shown / hidden / kept':>30s}   {'R2 PC1-5 mean shown / hidden / kept':>36s} | head shift | held-out gain")
    for name, kind, run in RUNS:
        ck = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
        data = Data(PATHS["train"], stats=ck["stats"])
        mod = hs_e2e if kind == "gru" else hs_fused_sde
        if kind == "fused":
            hs_fused_sde.prepare(data)
        model = mod.make_model(data, ck["args"])
        model.load_state_dict(ck["model"])
        model.eval()
        pats = np.array([i for i in range(data.n_pat) if data.rna_row[data.final_row(i)] >= 0])
        Y = pcs[data.rna_row[[data.final_row(i) for i in pats]]]
        folds = np.random.default_rng(0).integers(0, 5, len(pats))
        Zs, Fs = latents(kind, model, data, pats, False)
        Zh, Fh = latents(kind, model, data, pats, True)
        rs, rh = ridge_cv_r2(Zs, Y, folds), ridge_cv_r2(Zh, Y, folds)
        y = np.array([data.y[data.final_row(i)] for i in pats])
        ho = data.flag[pats]
        gain = np.abs(Fh - y)[ho].mean() - np.abs(Fs - y)[ho].mean()
        print(f"{name:40s} {rs[0]:8.2f} / {rh[0]:5.2f} / {rs[0] - rh[0]:+5.2f}      {rs.mean():12.2f} / {rh.mean():5.2f} / {rs.mean() - rh.mean():+5.2f}"
              f"      | {np.abs(Fs - Fh).mean():6.2f}     | {gain:+.3f} (n={ho.sum()})")
    print(f"patients with RNAseq before the cutoff: {len(pats)}")


if __name__ == "__main__":
    main()
