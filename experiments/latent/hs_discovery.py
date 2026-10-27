"""
Discovery tests on a saved fused model (hs_fused_sde.py; no training): what the model learned about RNAseq, ethnicity and images, each checked
against the raw data. Model effects are associations that a forecaster learned, not causal effects.

1. RNAseq PCs (PCA of the standardized RNAseq embeddings of all samples, no labels): the RNAseq input of each patient with a sample before the
   cutoff is shifted by +1 SD along PC k (every step that shows the sample); change of the 6-month forecast at the cutoff, patients on a
   biologic at the cutoff vs not. Raw-data check: Spearman of the PC score with the real label, and with the label minus the model's
   clinical-only forecast (the part of the outcome that the EHR does not explain).
2. Ethnicity: each held-out patient forecast with every ethnicity (the rest unchanged): mean change against the patient's own. Raw-data check:
   mean real label and mean (real label - 12-month visit mean) per ethnicity, all train-folder patients.
3. Image: at the first scored visit with a new image (held-out patients), the visit score is hidden from the input at that step; the decoder
   (with the visit type) predicts the score from the posterior mean, image shown vs hidden: MAE and R2 against the true score. Raw-data check:
   section 9.3 of the handoff (ridge on 20 PCs: history R2 0.655, image alone 0.550, both 0.732).

    python -m experiments.latent.hs_discovery [runs/hs_fused_sde_<stamp>] [--device cuda]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.latent.hs_e2e import PATHS, Data
from experiments.latent.hs_fused_sde import make_model, prepare

DEMO = "test_data/hs_public_dataset/public/train/demographics.csv"


def forecast(model, data, pats, device, **kw):
    """head forecast at the final step (score points); kw: pack options (hide_img, hide_rna)"""
    out = []
    with torch.no_grad():
        for a in range(0, len(pats), 100):
            sel = [int(i) for i in pats[a:a + 100]]
            mu = model.forecast(model.encode(data.pack(sel, device, **kw)))[0].cpu().numpy()
            out += [mu[j, data.n_rows(i) - 1] for j, i in enumerate(sel)]
    return np.array(out) * data.stats["ystd"] + data.stats["ymean"]


def rna_test(model, data, device):
    E = data.rna_emb.astype(np.float64)
    U, S, Vt = np.linalg.svd(E - E.mean(0), full_matrices=False)
    sd = S / np.sqrt(len(E))
    score = U * S / sd  # standardized PC scores of each sample
    nm = data.names.index
    pats = np.array([i for i in range(data.n_pat) if data.rna_row[data.final_row(i)] >= 0])
    fr = np.array([data.final_row(i) for i in pats])
    drugs = {d: data.raw[fr, nm(f"bio_active_{d}")] > 0 for d in ("adalimumab", "bimekizumab", "secukinumab")}
    y = data.y[fr]
    base = forecast(model, data, pats, device)
    clin = forecast(model, data, pats, device, hide_img=True, hide_rna=True)
    print(f"1. RNAseq PCs: {len(pats)} patients with a sample before the cutoff (all on a biologic at the cutoff: the sample is taken at the "
          f"start; {data.flag[pats].sum()} held-out); drug at the cutoff: " + ", ".join(f"{d} {m.sum()}" for d, m in drugs.items()))
    print("   PC  var%  | model: forecast change for +1 SD (score points)       | raw data: Spearman of the PC score with")
    print("             |   all            " + "  ".join(f"{d[:4]:>6s}" for d in drugs) + "   |   real label   label - clinical forecast (all / "
          + " / ".join(d[:4] for d in drugs) + ")")
    orig = data.rna_emb
    for k in range(5):
        data.rna_emb = (orig + sd[k] * Vt[k]).astype(np.float32)
        d = forecast(model, data, pats, device) - base
        data.rna_emb = orig
        s = score[data.rna_row[fr], k]
        res = y - clin
        print(f"   {k + 1:2d}  {100 * sd[k] ** 2 / (sd ** 2).sum():4.1f}  |   {d.mean():+.2f} (sd {d.std():.2f}) " + "  ".join(f"{d[m].mean():+6.2f}" for m in drugs.values())
              + f"   |   {spearmanr(s, y)[0]:+.2f}         {spearmanr(s, res)[0]:+.2f} / " + " / ".join(f"{spearmanr(s[m], res[m])[0]:+.2f}" for m in drugs.values()))


def ethnicity_test(model, data, device):
    demo = pd.read_csv(DEMO)
    cats = sorted(demo.ethnicity.dropna().unique())
    j = data.names.index("ethnicity")
    te = np.flatnonzero(data.flag)
    rows = np.concatenate([np.arange(data.off[i], data.off[i + 1]) for i in te])
    own = data.raw[[data.final_row(i) for i in te], j]
    has = data.M[[data.final_row(i) for i in te], j]
    oh = [data.names.index(f"ethnicity={c}") for c in range(len(cats))] if "ethnicity=0" in data.names else []
    cols = oh if oh else [j]  # one-hot columns (onehot runs) or the ordinal code
    X0, M0 = data.X[np.ix_(rows, cols)].copy(), data.Mf[np.ix_(rows, cols)].copy()
    F = np.zeros((len(te), len(cats)))
    for c in range(len(cats)):
        for q, jj in enumerate(cols):
            val = float(q == c) if oh else float(c)
            data.X[rows, jj] = (val - data.stats["mean"][jj]) / data.stats["std"][jj]
            data.Mf[rows, jj] = 1
        F[:, c] = forecast(model, data, te, device)
    data.X[np.ix_(rows, cols)], data.Mf[np.ix_(rows, cols)] = X0, M0
    print(f"\n   (input coding: {'one-hot' if oh else 'one ordinal code: effects are a slope on an arbitrary order'})")
    print(f"\n2. Ethnicity (held-out, {len(te)} patients): mean forecast with each assigned ethnicity minus the forecast with the White value")
    w = cats.index("White") if "White" in cats else 0
    for c, name in enumerate(cats):
        print(f"   assigned {name:35s} {np.mean(F[:, c] - F[:, w]):+.3f}")
    print("   raw data (all train-folder patients): n, mean real label, mean (real label - 12-month visit mean)")
    allp = np.arange(data.n_pat)
    fr = np.array([data.final_row(i) for i in allp])
    eth = np.where(data.M[fr, j], data.raw[fr, j], -1)
    m12 = data.names.index("mean_12m")
    ok12 = data.M[fr, m12]
    for c, name in enumerate(cats):
        s = eth == c
        r = (data.y[fr] - data.raw[fr, m12])[s & ok12]
        print(f"   {name:35s} n={s.sum():5d}  label {data.y[fr][s].mean():6.2f}  label - 12-month mean {r.mean():+.2f} (n={len(r)})")
    _ = own, has


def image_test(model, data, device):
    own = data.names.index("own_score")
    te = np.flatnonzero(data.flag)
    rows, pats = [], []
    for i in te:
        r = np.flatnonzero(data.img_new[data.off[i]:data.off[i + 1]] & data.has_score[data.off[i]:data.off[i + 1]])
        if len(r):
            rows.append(data.off[i] + r[0])
            pats.append(i)
    rows, pats = np.array(rows), np.array(pats)
    X0, M0 = data.X[rows, own].copy(), data.Mf[rows, own].copy()
    data.X[rows, own], data.Mf[rows, own] = 0, 0
    true = data.score[rows]
    res = {}
    for hide in (False, True):
        P = np.zeros(len(rows))
        with torch.no_grad():
            for a in range(0, len(pats), 100):
                sel = [int(i) for i in pats[a:a + 100]]
                model.encode(data.pack(sel, device, hide_img=hide))
                m = model.last_post[0]
                k = torch.as_tensor([rows[a + b] - data.off[i] for b, i in enumerate(sel)], device=device)
                mk = m[torch.arange(len(sel), device=device), k]
                vt = torch.as_tensor(data.vt[rows[a:a + len(sel)]], device=device)
                mu = model.decode(mk, vt)[0].cpu().numpy()
                P[a:a + len(sel)] = np.expm1(mu * data.stats["ls"] + data.stats["lm"])
        res[hide] = P
    data.X[rows, own], data.Mf[rows, own] = X0, M0
    print(f"\n3. Image (held-out, first scored visit with a new image, n={len(rows)}): the visit score hidden at that step; decoder with the visit type")
    for hide, name in ((True, "history only (image hidden)"), (False, "history + image")):
        P = res[hide]
        r2 = 1 - ((true - P) ** 2).sum() / ((true - true.mean()) ** 2).sum()
        print(f"   {name:30s} MAE {np.abs(P - true).mean():.2f}  R2 {r2:.3f}")
    print("   raw data (section 9.3, ridge on 20 PCs, 5-fold): history R2 0.655, image alone 0.550, both 0.732")
    # does the image hold information beyond the latent? ridge (5-fold) from [model prediction without the image, 20 image PCs] to the score
    E = data.img_emb.astype(np.float64)
    U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    Z = (U[:, :20] * S[:20])[data.img_row[rows]]
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    for feats, name in ((res[True][:, None], "model without image"), (np.column_stack([res[True], Z]), "model without image + 20 image PCs"),
                        (Z, "20 image PCs alone")):
        P = np.zeros(len(rows))
        for f in range(5):
            tr, ts = folds != f, folds == f
            mu, sdv = feats[tr].mean(0), feats[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(tr.sum()), (feats[tr] - mu) / sdv])
            W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ true[tr])
            P[ts] = np.column_stack([np.ones(ts.sum()), (feats[ts] - mu) / sdv]) @ W
        print(f"   second stage, ridge 5-fold: {name:38s} R2 {1 - ((true - P) ** 2).sum() / ((true - true.mean()) ** 2).sum():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/hs_fused_sde_20261026-190817")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="", help="comma-separated subset of rna,ethnicity,image")
    args = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(f"{args.run}/model.pt", map_location=args.device, weights_only=False)
    data = Data(PATHS["train"], stats=ck["stats"])
    prepare(data)
    model = make_model(data, ck["args"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"model {args.run}")
    tests = {"rna": rna_test, "ethnicity": ethnicity_test, "image": image_test}
    for k, f in tests.items():
        if not args.only or k in args.only.split(","):
            f(model, data, args.device)


if __name__ == "__main__":
    main()
