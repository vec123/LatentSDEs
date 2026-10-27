"""
Head 1 and head 2 of an end-to-end run (hs_e2e.py) on the held-out patients with an open biologic line at the cutoff (all, and those
with an RNAseq sample), with the image and the RNAseq hidden or not. MAE against the real label of:
  head 1 at the final step;
  head 2 at the final step with the current drug, horizon 0 (x a, the scale factor of hs_task2.py);
  head 2 at the decision step of the current line with its drug, horizon = cutoff - decision day (x a);
and the mean change of the untreated forecast Y0 (head 2, plan none, from the decision step) against the full input.
Writes <run>/heads_preds.npz (read by xgb_start.py).

    python -m experiments.latent.heads_rna runs/hs_e2e_<stamp>
"""
import argparse

import numpy as np
import torch

from experiments.latent.hs_task2 import counterfactual, load, open_lines, scale_factor

SETTINGS = {"clin": ("clinical only", dict(hide_img=True, hide_rna=True)), "img": ("clinical + image (RNAseq hidden)", dict(hide_rna=True)),
            "full": ("clinical + image + RNAseq", {})}
HEADS = {"h1": "head 1 (final step)", "h2cut": "head 2 (final step, current drug)", "h2start": "head 2 (decision step, given drug)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, ck = load(args.run, device)
    a, _ = scale_factor(model, data, list(np.flatnonzero(~data.flag)), device)
    lines = open_lines(data, np.flatnonzero(data.flag))
    y = np.array([data.y[data.off[l[0]] + l[4]] for l in lines])
    rna = np.array([data.rna_row[data.off[l[0]] + l[4]] >= 0 for l in lines])
    drug, n = np.array([l[2] for l in lines]), np.arange(len(lines))
    cf = {s: counterfactual(model, data, lines, device, **kw) for s, (_, kw) in SETTINGS.items()}
    P = {f"{s}_{h}": v for s, c in cf.items() for h, v in (("h1", c["y1"]), ("h2cut", a * c["y1_h2"]), ("h2start", a * c["yd"][n, drug]))}
    np.savez(f"{args.run}/heads_preds.npz", patient_id=data.pid[[l[0] for l in lines]], y=y, has_rna=rna, a=a, **P)
    rng = np.random.default_rng(0)
    for gname, g in (("held-out patients with an open biologic line", np.ones(len(y), bool)), ("of them with an RNAseq sample", rna)):
        print(f"\n{gname}: n={g.sum()}; scale factor a = {a:.3f}")
        print(f"{'input':34s} " + " ".join(f"{h:>36s}" for h in HEADS.values()) + f" {'mean |log Y0 / Y0 full|':>24s}")
        for s, (label, _) in SETTINGS.items():
            print(f"{label:34s} " + " ".join(f"{np.mean(np.abs(P[f'{s}_{h}'][g] - y[g])):36.3f}" for h in HEADS)
                  + f" {np.mean(np.abs(np.log(np.maximum(cf[s]['y0'][g], 0.5) / np.maximum(cf['full']['y0'][g], 0.5)))):24.3f}")
        print("gain of the RNAseq (MAE with the RNAseq hidden - MAE with it), 95% interval over patients (2,000 draws)")
        for h, label in HEADS.items():
            e = np.abs(P[f"img_{h}"][g] - y[g]) - np.abs(P[f"full_{h}"][g] - y[g])
            m = e[rng.integers(0, len(e), (2000, len(e)))].mean(1)
            print(f"  {label:36s} {e.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
