"""
Parameter recovery test of flare_model.py (CPU, ~2 min): can the fit find the flare-amplitude variance and the patient variation of
amplitude and duration at these visit times? Truth: the fitted model of <run> (baseline = its posterior mean, onsets = the observed flare
visits, its drug and feature effects), with the sd of log A = 0.8 and patient sds 0.25 (amplitude, duration); new scores at the scored
visits with Laplace noise 0.155 (the near-replicate value of flare_facts.py). Then a new fit (same settings) on the simulated scores.
If the recovered variances collapse here too, the inference is the cause, not the data.

    python -m experiments.latent.flare_recovery [runs/flare_model_rw2] [iters]
"""
import sys

import numpy as np
import torch

from experiments.latent import flare_model as fm

S_A, S_AMP, S_DUR, SIG_Y = 0.8, 0.25, 0.25, 0.155


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/flare_model_rw2"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    nc = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False  # the form of the fit (the truth is the same)
    torch.set_num_threads(4)
    torch.manual_seed(1)
    d = fm.load()
    ck = torch.load(f"{run}/model.pt", weights_only=False)
    truth = fm.FlareModel(d, rw=ck["rw"], nc=ck.get("nc", False))
    truth.load_state_dict(ck["model"])
    with torch.no_grad():
        th = truth.split(truth.th_m)
        n, K = len(d["pats"]), len(d["f_t"])
        c = d["X"] @ th["W"].T + torch.tensor([0.0, S_AMP, S_DUR]) * torch.randn(n, 3)
        c[:, 0] = truth.parts(d, sample=False)["c"][:, 0]  # the onsets are the observed ones: the frequency code stays the fitted one
        V = torch.cat([d["f_u"], d["f_ab"][:, None]], 1) - truth.v_mean
        bio = (d["f_u"].sum(1) > 0).float()
        rna = d["R"] @ th["g_R"]
        fp = d["f_p"]
        logA = truth.mu_A + c[fp, 1] + V @ th["b_A"] + bio * rna[fp, 1] + S_A * torch.randn(K)
        logD = truth.d0 + c[fp, 2] + V @ th["b_D"] + bio * rna[fp, 2]
        F = torch.zeros_like(d["o_y"]).index_add_(0, d["q_j"], (logA[d["q_k"]] - d["q_lag"] / logD[d["q_k"]].exp()).exp())
        b = (1 - d["o_f"]) * truth.k_m[d["o_k"]] + d["o_f"] * truth.k_m[d["o_k"] + 1] + d["o_u"] @ th["b_L"]
        y = b + F + torch.distributions.Laplace(0.0, SIG_Y).sample(b.shape)
    d2 = dict(d, o_y=y)
    model = fm.FlareModel(d2, rw=ck["rw"], nc=nc)
    fm.fit(model, d2, iters, print)
    with torch.no_grad():
        r = model.parts(d2, sample=False)
    cm = r["c"]
    print(f"\nfit form: {'non-centred' if nc else 'centred'}")
    corr = lambda a, b_: float(np.corrcoef(a.numpy(), b_.numpy())[0, 1])
    print(f"\nrecovery (truth -> fitted on the simulated scores):")
    print(f"   sd of log A          {S_A:.3f} -> {model.ls_A.exp():.3f}")
    print(f"   patient sd amplitude {S_AMP:.3f} -> {model.ls_c.exp()[1]:.3f}   corr(true c_amp, posterior mean) {corr(c[:, 1], cm[:, 1]):.2f}")
    print(f"   patient sd duration  {S_DUR:.3f} -> {model.ls_c.exp()[2]:.3f}   corr(true c_dur, posterior mean) {corr(c[:, 2], cm[:, 2]):.2f}")
    print(f"   visit noise          {SIG_Y:.3f} -> {model.ls_y.exp():.3f}")
    print(f"   half-life (days)     {np.log(2) * truth.d0.exp():.1f} -> {np.log(2) * model.d0.exp():.1f}")
    print(f"   median amplitude     x{np.exp(np.exp(float(truth.mu_A))):.3f} -> x{np.exp(np.exp(float(model.mu_A))):.3f}")
    print(f"   corr(true log A, posterior mean) {corr(logA, r['logA']):.2f}")


if __name__ == "__main__":
    main()
