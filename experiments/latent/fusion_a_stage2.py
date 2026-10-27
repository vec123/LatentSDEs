"""
Option A (fusion after the GRU) as a stage-2 test on a frozen end-to-end run (hs_e2e.py): the forecast is head 1 with the image and
the RNAseq hidden (the clinical forecast) plus a correction from the GRU state and the modalities, sigmoid(gate) x MLP (no hidden
layer: linear). Patients: those with an image at the cutoff (every RNAseq patient has one). The correction is fitted twice and scored
on the held-out image patients of the train folder, against the run's own fusion (modalities at the GRU input) and XGBoost:
  E1: fitted on the final steps of the training patients (in-sample for the GRU), early stopping on its validation patients;
  E2: 5-fold cross-validation over the validation and held-out patients (the GRU never trained on their labels).
The states are cached in <run>/stage2_states.npz (the GPU part runs once).

    python -m experiments.latent.fusion_a_stage2 runs/hs_e2e_<stamp> [--xgb runs/xgb_e2e_tuned]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from experiments.latent.hs_e2e import DRUGS
from experiments.latent.hs_task2 import load


@torch.no_grad()
def extract(model, data, pats, device, chunk=50):
    """per patient at the final step: the state and head 1 with the modalities hidden, head 1 with all modalities (label units)"""
    ys, ym = data.stats["ystd"], data.stats["ymean"]
    order = np.argsort([data.n_rows(i) for i in pats])
    H, clin, full = np.zeros((len(pats), model.gru.hidden_size), np.float32), np.zeros(len(pats)), np.zeros(len(pats))
    for a in range(0, len(order), chunk):
        js = order[a:a + chunk]
        sel = [pats[j] for j in js]
        b = torch.arange(len(sel), device=device)
        k = torch.as_tensor([data.final_row(i) - data.off[i] for i in sel], device=device)
        h = model.encode(data.pack(sel, device, hide_img=True, hide_rna=True))[b, k]
        H[js], clin[js] = h.cpu().numpy(), (model.forecast(h)[0] * ys + ym).cpu().numpy()
        full[js] = (model.forecast(model.encode(data.pack(sel, device))[b, k])[0] * ys + ym).cpu().numpy()
    return H, clin, full


class Correction(nn.Module):
    def __init__(self, d, hid):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1)) if hid else nn.Linear(d, 1)
        last = self.m[-1] if hid else self.m
        nn.init.zeros_(last.weight)  # the fit starts at the base forecast
        nn.init.zeros_(last.bias)
        self.g = nn.Linear(d, 1) if hid else None

    def forward(self, x):
        r = self.m(x)[:, 0]
        return r if self.g is None else torch.sigmoid(self.g(x)[:, 0]) * r


def fit(Z, base, y, tr, va, hid, seed, steps=1500, every=10, patience=30, wd=1e-3):
    """the correction r in y ~ base + r(Z): L1, full batch, the check with the lowest MAE on va; r for all rows"""
    torch.manual_seed(seed)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    Zs, t = torch.as_tensor((Z - mu) / sd, dtype=torch.float32), torch.as_tensor(y - base, dtype=torch.float32)
    net = Correction(Z.shape[1], hid)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
    best, best_step, state = np.inf, 0, None
    for s in range(steps):
        if s % every == 0:
            net.eval()
            with torch.no_grad():
                v = (net(Zs[va]) - t[va]).abs().mean().item()
            if v < best:
                best, best_step, state = v, s, {k: x.clone() for k, x in net.state_dict().items()}
            elif s - best_step > patience * every:
                break
        net.train()
        loss = (net(Zs[tr]) - t[tr]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        return net(Zs).numpy()


def cv(Z, base, y, pool, hid, seeds, k=5):
    """out-of-fold corrections by k-fold cross-validation over pool (each training part: 80% fit, 20% early stopping)"""
    out, rng = np.full(len(y), np.nan), np.random.default_rng(0)
    folds = np.array_split(rng.permutation(pool), k)
    for f in range(k):
        rest = rng.permutation(np.concatenate([folds[g] for g in range(k) if g != f]))
        n = int(0.8 * len(rest))
        out[folds[f]] = np.mean([fit(Z, base, y, rest[:n], rest[n:], hid, s)[folds[f]] for s in range(seeds)], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--xgb", default="runs/xgb_e2e_tuned")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cache = f"{args.run}/stage2_states.npz"
    if not os.path.exists(cache):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, data, ck = load(args.run, device)
        a = ck["args"]
        pool = np.random.default_rng(a.get("seed", 0)).permutation(np.flatnonzero(~data.flag))  # the split of hs_e2e.py
        split = np.zeros(data.n_pat, np.int64)  # 0 training, 1 validation, 2 held-out
        split[pool[:int(a.get("val_frac", 0.1) * len(pool))]], split[data.flag] = 1, 2
        fr = np.array([data.final_row(i) for i in range(data.n_pat)])
        pats = np.flatnonzero(data.img_row[fr] >= 0)
        H, clin, full = extract(model, data, list(pats), device)
        f, ri = fr[pats], data.rna_row[fr[pats]]
        np.savez(cache, patient_id=data.pid[pats], split=split[pats], y=data.y[f], H=H, clin=clin, full=full,
                 img=data.img_emb[data.img_row[f]], rna=np.where((ri >= 0)[:, None], data.rna_emb[np.clip(ri, 0, None)], 0),
                 has_rna=(ri >= 0).astype(np.float32),
                 drug=np.nan_to_num(np.column_stack([data.raw[f, data.names.index(f"bio_active_{d.lower()}")] for d in DRUGS])))
        print(f"cached the states of {len(pats)} image patients in {cache}", flush=True)
        del model, data
    torch.set_num_threads(2)
    z = np.load(cache)
    split, y, clin, full, has_rna = z["split"], z["y"].astype(np.float64), z["clin"], z["full"], z["has_rna"] > 0
    ho = split == 2
    g = np.load(f"{args.run}/final_preds.npz")
    pos = {p: i for i, p in enumerate(g["patient_id"])}
    print(f"check: |full - final_preds.npz| max {np.abs(full[ho] - g['full_mu'][[pos[p] for p in z['patient_id'][ho]]]).max():.2e}")
    for s, nme in enumerate(("training (in-sample)", "validation", "held-out")):
        m = split == s
        print(f"  {nme:22s} n={m.sum():5d}  MAE clinical {np.mean(np.abs(clin[m] - y[m])):.3f}  run's fusion {np.mean(np.abs(full[m] - y[m])):.3f}")
    xg = np.load(f"{args.xgb}/final_preds.npz")
    xpos = {p: i for i, p in enumerate(xg["patient_id"])}
    xgb = np.array([xg["full_mu"][xpos[p]] if p in xpos else np.nan for p in z["patient_id"]])

    Ze = np.column_stack([z["img"], z["rna"], z["has_rna"], z["drug"]])
    Zhe = np.column_stack([z["H"], Ze])
    variants = {"linear on the modalities": (Ze, 0, clin), "option A: gated MLP on state + modalities": (Zhe, 32, clin),
                "gated MLP on top of the run's fusion": (Zhe, 32, full)}
    tr, va, pool = np.flatnonzero(split == 0), np.flatnonzero(split == 1), np.flatnonzero(split > 0)
    rows = {"clinical (head 1, modalities hidden)": clin, "run's fusion (modalities at the GRU input)": full, "XGBoost": xgb}
    for name, (Z, hid, base) in variants.items():
        rows[f"E1 {name}"] = base + np.mean([fit(Z, base, y, tr, va, hid, s) for s in range(args.seeds)], 0)
        rows[f"E2 {name}"] = base + cv(Z, base, y, pool, hid, args.seeds)
        print(f"  fitted: {name}", flush=True)

    rng = np.random.default_rng(0)
    groups = {"held-out with image (ex 3a)": ho, "held-out with image + RNAseq (ex 3b)": ho & has_rna}
    print(f"\nMAE at the cutoff; difference to the run's fusion with a 95% interval over patients (2,000 draws); * better, ! worse")
    for gname, s in groups.items():
        print(f"\n{gname}: n={s.sum()}")
        for name, p in rows.items():
            e = np.abs(p[s] - y[s]) - np.abs(full[s] - y[s])
            m = e[rng.integers(0, s.sum(), (2000, s.sum()))].mean(1)
            lo, hi = np.percentile(m, [2.5, 97.5])
            mark = "  *" if hi < 0 else ("  !" if lo > 0 else "")
            print(f"  {name:52s} {np.mean(np.abs(p[s] - y[s])):6.3f}   {e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}")


if __name__ == "__main__":
    main()
