"""
Extends the prediction of a no-code run of run_hs_seq by 12 months with its latent SDE (no training, CPU).

At a cut visit the posterior state (it has read the visits up to the cut only) is sampled S times and the prior SDE dz = f(z) dt + g dW is simulated
for 12 months (Euler-Maruyama, `sub` sub-steps per month); at each month h the decoder gives the predicted mean score of the 6 months
after cut + h. Band: +- 1 std of the total, sqrt(variance of the decoded means over the paths + mean decoder variance); at h = 0 this is the decoder
std alone, the band of plot_cuts.py. The true value at cut + h is the mean of the scores in (cut + h, cut + h + 6] (the definition of y).
The drift of these runs was only a regularizer in training (it was never trained to forecast), so the script also scores the rollout:
for random test cuts, the RMSE at h = 0, 3, 6, 9, 12 of the rollout mean, of the cut prediction kept flat, and of the past 6-month mean.

    python -m experiments.latent.plot_rollout runs/hs_seq_<stamp> --dim_z 32 --dec_hidden 128 128 [--extra_features] --cpu
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.plot_rows import BLUE, GRID, INK, MUTED, ORANGE, SURFACE
from experiments.latent.run import build_model

HORIZON, WINDOW = 12, 6.0  # months


@torch.no_grad()
def rollout(model, data, i, k, device, S=64, sub=4):
    """Decoded mean and std (original units), shape (S, HORIZON + 1), of S simulated paths from the posterior state at row k of patient i."""
    code = torch.zeros(1, 0, device=device)
    batch = data.pack([(int(i), k + 1, None)], code, device)  # the first k + 1 rows only: nothing later is read
    model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
    m, logv, _, _ = model.posterior(batch["t"][:, -1:])
    z = m[0, 0] + (0.5 * logv[0, 0]).exp() * torch.randn(S, m.shape[-1], device=device)
    mu = torch.zeros(S, 0, device=device)
    dt = (1.0 / R.T_SCALE) / sub  # one month is 1 / T_SCALE in the model's time unit
    means, stds = [], []
    for h in range(HORIZON + 1):
        dist = model.decoder(z, mu)
        means.append(dist.mean[:, 0] * data.ystd + data.ymean)
        stds.append(dist.stddev[:, 0] * data.ystd)
        if h < HORIZON:
            for _ in range(sub):
                z = z + model.drift(z, mu, mu) * dt + model.diffusion(z, mu, None) * dt ** 0.5 * torch.randn_like(z)
    return torch.stack(means, 1).cpu().numpy(), torch.stack(stds, 1).cpu().numpy()


def true_curve(t, own, cut):
    """True mean score of (cut + h, cut + h + 6] for h = 0..12 (nan: no score in the window, or the window ends after the last visit)."""
    out = np.full(HORIZON + 1, np.nan)
    for h in range(HORIZON + 1):
        lo = cut + h
        if lo + WINDOW > t[-1]:
            continue
        w = (t > lo) & (t <= lo + WINDOW) & np.isfinite(own)
        if w.any():
            out[h] = own[w].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--extra_features", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--min_rows", type=int, default=20)
    ap.add_argument("--min_history", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(0)
    data = R.Data(R.LIGHT, extra=args.extra_features)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    model = build_model(R.CFG, None, 1, None).to(device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device)["model"])
    model.eval()

    # ---- the same patients and cuts as plot_cuts.py (same seed, same draws)
    rng = np.random.default_rng(args.seed)
    pool = [i for i in data.test_idx if data.ym[data.off[i]:data.off[i + 1]].sum() >= args.min_rows]
    pick = rng.choice(pool, args.n, replace=False)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "font.size": 9})
    cols = 5
    fig, axes = plt.subplots(-(-args.n // cols), cols, figsize=(4.0 * cols, 3.1 * -(-args.n // cols)))
    grid = np.arange(HORIZON + 1.0)
    for ax, i in zip(axes.ravel(), pick):
        s = slice(data.off[i], data.off[i + 1])
        t, ok = data.months[s], data.ym[s]
        own = np.where(data.raw["own_score"][1][s], data.raw["own_score"][0][s], np.nan)
        cand = np.flatnonzero(ok & (np.arange(len(t)) >= args.min_history))
        k = int(rng.choice(cand))
        tc = t[k]
        mean, std = rollout(model, data, i, k, device)
        centre = mean.mean(0)
        total = np.sqrt(mean.var(0) + (std ** 2).mean(0))
        past, future = np.arange(len(t)) <= k, np.arange(len(t)) > k
        ax.axvline(tc, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.plot(t[past], own[past], color=MUTED, lw=0.8, alpha=0.7, zorder=2)
        ax.scatter(t[past], own[past], s=12, color=MUTED, linewidths=0, label="scores up to the cut (read by the model)", zorder=3)
        ax.scatter(t[future & (t <= tc + 18.0)], own[future & (t <= tc + 18.0)], s=22, facecolor="none", edgecolor=MUTED,
                   linewidths=0.9, label="scores after the cut (not read)", zorder=3)
        ax.fill_between(tc + grid, centre - total, centre + total, color=BLUE, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(tc + grid, centre, color=BLUE, lw=2.0, label="SDE rollout: prediction of the next-6-month mean at each month", zorder=6)
        ax.plot(tc + grid, np.full_like(grid, centre[0]), color=BLUE, lw=1.0, ls=":", label="cut prediction kept flat", zorder=5)
        ax.plot(tc + grid, true_curve(t, own, tc), color=INK, lw=1.8, label="true next-6-month mean at each month", zorder=5)
        ax.set_xlim(max(0.0, tc - 30.0), tc + 18.0)
        ax.set_title(f"{data.pid[i]}  cut at month {tc:.0f}:  h=0 {centre[0]:.1f}  h=12 {centre[-1]:.1f}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("month")
        ax.grid(axis="y", color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    h_, l_ = axes.ravel()[0].get_legend_handles_labels()
    order = [2, 3, 4, 0, 1]  # prediction and truth first
    fig.legend([h_[j] for j in order], [l_[j] for j in order], loc="upper center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.998))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{args.run}/rollout_12_months.png"
    fig.savefig(out, dpi=105)
    print("saved", out)

    # ---- does the rollout forecast? RMSE by horizon on random test cuts
    rng = np.random.default_rng(123)
    rows = []
    for i in rng.choice(pool, args.n_eval, replace=False):
        s = slice(data.off[i], data.off[i + 1])
        cand = np.flatnonzero(data.ym[s] & (np.arange(data.n_rows(i)) >= args.min_history))
        if len(cand):
            rows.append((int(i), int(rng.choice(cand))))
    err = {"SDE rollout": [[] for _ in grid], "flat (cut prediction)": [[] for _ in grid], "past 6-month mean": [[] for _ in grid]}
    nonfinite = 0
    for i, k in rows:
        s = slice(data.off[i], data.off[i + 1])
        t = data.months[s]
        own = np.where(data.raw["own_score"][1][s], data.raw["own_score"][0][s], np.nan)
        truth = true_curve(t, own, t[k])
        mean, _ = rollout(model, data, i, k, device, S=32)
        centre = mean.mean(0)
        if not np.isfinite(centre).all():
            nonfinite += 1
            continue
        base = data.raw["mean_6m"][0][s][k] if data.raw["mean_6m"][1][s][k] else data.train_mean_y
        for h in range(HORIZON + 1):
            if np.isfinite(truth[h]):
                err["SDE rollout"][h].append(centre[h] - truth[h])
                err["flat (cut prediction)"][h].append(centre[0] - truth[h])
                err["past 6-month mean"][h].append(base - truth[h])
    print(f"\nRMSE by horizon over {len(rows)} random test cuts ({nonfinite} rollouts not finite, left out):")
    print(f"{'h (months)':24s}" + "".join(f"{h:>8d}" for h in (0, 3, 6, 9, 12)) + "     n at h=0 / 12")
    for name, e in err.items():
        print(f"{name:24s}" + "".join(f"{np.sqrt(np.mean(np.square(e[h]))):8.3f}" for h in (0, 3, 6, 9, 12))
              + f"     {len(e[0])} / {len(e[12])}")


if __name__ == "__main__":
    main()
