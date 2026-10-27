"""
Does the GRU's predicted drug difference hold where the data allow a check without confounding? Line 2 starts of Bimekizumab or Secukinumab (the
near-random choice, propensity AUC 0.505) with the outcome of causal_modifiers.py (minus the ratio of the mean score days 90..270 after the start to
the mean of the 180 days before, pre >= 3, capped at 3; higher = better).

GRU: head 2 at the decision step (the day before the start), window months 3..9 (horizon 3, months on the drug 3), visit scale, plans Bimekizumab,
Secukinumab and none, from the cross-validation fold model that held the patient out (hs_e2e.py --fold f/5 --drop ethnicity, runs/cv5_runs.txt).
tau_hat = -(Y_Bime - Y_Secu) / pre mean: the predicted effect of Bimekizumab minus Secukinumab in the units of the AIPW scores Gamma of
causal_modifiers.py (the same cross-fitted ridge outcome models and logistic propensity, no GRU input).
Tests: the average (mean tau_hat against mean Gamma); the best linear projection Gamma ~ 1 + (tau_hat - mean tau_hat) (slope 1: the predicted
differences have the right size, 0: no information), robust (HC1) errors and a bootstrap; Gamma per quarter of tau_hat; the rank correlation. The
factual side: the predicted ratio under the drug given against the observed ratio.

    ../environments/torch_gpu_env/Scripts/python.exe -m experiments.latent.causal_gru_check [runs/cv5_runs.txt]
"""
import sys
from math import erf, sqrt

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.latent import causal_modifiers as CM
from experiments.latent.hs_e2e import plan_at
from experiments.latent.hs_task2 import load

HZ = 3.0  # months from the decision step to the window start: window months 3..9, about days 90..270 after the start
rng = np.random.default_rng(0)


@torch.no_grad()
def gru_preds(runs, need, device):
    """need: patient -> decision rows; returns row -> (Y_none, Y_Bime, Y_Secu, fold), each from the fold model that held the patient out"""
    out = {}
    for f, run in sorted(runs.items()):
        model, data, _ = load(run, device)
        pats = sorted(i for i in need if data.flag[i])
        for a in range(0, len(pats), 50):
            sel = pats[a:a + 50]
            h = model.encode(data.pack(sel, device))
            bi, ki, rr = [], [], []
            for b, i in enumerate(sel):
                for r in need[i]:
                    bi.append(b), ki.append(r - data.off[i]), rr.append(r)
            t = lambda v, tp=torch.float32: torch.as_tensor(np.asarray(v), dtype=tp, device=device)
            hz = t(np.full(len(rr), HZ))
            ys = []
            for plan, tau in ((0, 0.0), (2, HZ), (3, HZ)):  # plan index = drug index + 1 (none, Ada, Bime, Secu)
                mu, _ = plan_at(model, h, t(bi, torch.long), t(ki, torch.long), hz, t(np.full(len(rr), tau)), t(np.full(len(rr), plan), torch.long))
                ys.append((mu * data.stats["ystd"] + data.stats["ymean"]).cpu().numpy())
            for j, r in enumerate(rr):
                out[r] = (ys[0][j], ys[1][j], ys[2][j], f)
        print(f"fold {f}: {sum(len(need[i]) for i in pats)} line 2 starts predicted ({run})", flush=True)
    return out


def blp(G, z):
    """Gamma ~ 1 + z: coefficients and HC1 standard errors"""
    D = np.column_stack([np.ones(len(z)), z])
    Ginv = np.linalg.inv(D.T @ D)
    b = Ginv @ D.T @ G
    res = G - D @ b
    V = Ginv @ (D.T * res ** 2) @ D @ Ginv * len(z) / (len(z) - 2)
    return b, np.sqrt(np.diag(V))


def main():
    listing = sys.argv[1] if len(sys.argv) > 1 else "runs/cv5_runs.txt"
    runs = {int(p[1]): p[5] for p in (ln.split() for ln in open(listing)) if len(p) >= 6 and p[0] == "fold" and p[3] == "0"}
    assert sorted(runs) == [0, 1, 2, 3, 4], runs
    R = CM.build(90, 270, 3)
    G, A, y = CM.aipw(R)
    rows, pat = np.array([x[0] for x in R]), np.array([x[1] for x in R])
    need = {}
    for r, i in zip(rows, pat):
        need.setdefault(int(i), []).append(int(r))
    P = gru_preds(runs, need, "cuda" if torch.cuda.is_available() else "cpu")
    y0, yb, ys = (np.array([P[int(r)][k] for r in rows]) for k in range(3))
    def pre_mean(r, i):  # the denominator of the ratio: mean visit score of the 180 days before the start
        sd, sv = CM.scores(i)
        dd = sd - (CM.day[r] + 1)
        return sv[(dd >= -180) & (dd < 0)].mean()

    pre = np.array([pre_mean(int(r), int(i)) for r, i in zip(rows, pat)])
    tau = -(yb - ys) / pre
    n = len(G)
    print(f"\n{n} line 2 starts with the outcome (Bimekizumab {A.sum()}, Secukinumab {n - A.sum()}); each GRU prediction from the fold model that "
          f"held the patient out\n")

    # the factual side: the predicted ratio under the drug given
    ratio_obs = -y
    ratio_hat = np.where(A == 1, yb, ys) / pre
    print("factual side: predicted ratio (drug given) against the observed ratio (capped at 3)")
    print(f"   MAE {np.mean(np.abs(np.minimum(ratio_hat, 3) - ratio_obs)):.3f} (predicting the mean ratio of the others: "
          f"{np.mean(np.abs(ratio_obs.mean() - ratio_obs)):.3f}); Spearman {spearmanr(ratio_hat, ratio_obs)[0]:+.3f}; "
          f"AUC for good response (ratio < 0.5) {roc_auc_score(ratio_obs < 0.5, -ratio_hat):.3f}")
    print(f"   predicted drug against no biologic (ratio units): mean {np.mean((np.where(A == 1, yb, ys) - y0) / pre):+.3f}")

    # the average effect
    bs = rng.integers(0, n, (2000, n))
    lo, hi = np.percentile(G[bs].mean(1), [2.5, 97.5])
    print(f"\naverage effect, Bimekizumab - Secukinumab on minus the ratio (> 0: Bimekizumab better)")
    print(f"   GRU predicted: mean tau_hat {tau.mean():+.4f} (sd over starts {tau.std():.4f}; Bimekizumab predicted better for {np.mean(tau > 0):.0%})")
    print(f"   AIPW estimate: mean Gamma {G.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # heterogeneity: best linear projection and quarters
    z = tau - tau.mean()
    b, se = blp(G, z)
    slopes = []
    for k in bs[:1000]:
        slopes.append(blp(G[k], tau[k] - tau[k].mean())[0][1])
    slo, shi = np.percentile(slopes, [2.5, 97.5])
    p = 2 * (1 - 0.5 * (1 + erf(abs(b[1] / se[1]) / sqrt(2))))
    print(f"\nheterogeneity: best linear projection Gamma ~ 1 + (tau_hat - mean)")
    print(f"   slope {b[1]:+.3f}, HC1 95% [{b[1] - 1.96 * se[1]:+.3f}, {b[1] + 1.96 * se[1]:+.3f}], p {p:.3f}; bootstrap [{slo:+.3f}, {shi:+.3f}]")
    print(f"   (slope 1: the predicted differences have the right size; 0: they carry no information about who benefits from which drug)")
    print(f"   smallest slope the test finds with 80% power: {2.8 * se[1]:.3f}; Spearman(tau_hat, Gamma) {spearmanr(tau, G)[0]:+.3f}")
    q = np.quantile(tau, [0.25, 0.5, 0.75])
    grp = np.digitize(tau, q)
    print("\n   AIPW effect per quarter of the GRU's predicted difference (Bimekizumab - Secukinumab, > 0: Bimekizumab better)")
    for k in range(4):
        s = grp == k
        print(f"   quarter {k + 1}: tau_hat {tau[s].min():+.3f}..{tau[s].max():+.3f} (mean {tau[s].mean():+.3f}), n={s.sum()}: AIPW {G[s].mean():+.3f} "
              f"+- {1.96 * G[s].std() / np.sqrt(s.sum()):.3f}")
    d41 = [G[k][grp[k] == 3].mean() - G[k][grp[k] == 0].mean() for k in bs[:1000]]
    print(f"   quarter 4 - quarter 1: AIPW {G[grp == 3].mean() - G[grp == 0].mean():+.3f} [{np.percentile(d41, 2.5):+.3f}, {np.percentile(d41, 97.5):+.3f}]; "
          f"predicted {tau[grp == 3].mean() - tau[grp == 0].mean():+.3f}")
    # policy: give the drug the GRU predicts as better, against the drug given at random (AIPW value)
    pol = np.where(tau > 0, G, -G) / 2  # value of the policy minus the coin flip, from the AIPW scores
    pv = [pol[k].mean() for k in bs[:1000]]
    print(f"\n   policy 'give the drug the GRU predicts as better' minus a coin flip: {pol.mean():+.4f} [{np.percentile(pv, 2.5):+.4f}, "
          f"{np.percentile(pv, 97.5):+.4f}] (minus the ratio)")


if __name__ == "__main__":
    main()
