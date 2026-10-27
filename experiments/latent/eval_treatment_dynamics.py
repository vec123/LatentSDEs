"""
Does the drift of a --treatment_u run use the treatment? (no training, CPU)

For every biologic start of the test patients that follows a labelled visit k (Biologic_ever goes up between visit k and k + 1), the posterior
state at visit k (it has read the visits up to k only) is rolled forward 12 months under three treatment paths, with the same noise in all three:
  hold     : the treatment state of visit k stays as it is (no change)
  treated  : the treatment state of visit k + 1 (the biologic has started) stays from the cut on
  realized : the path the patient really had (zero-order hold of the later visits; an oracle on the treatment, not a forecast)
At each month h the decoder gives the predicted mean score of the 6 months after the cut + h; the truth is that mean computed from the scores.
Reported: the RMSE by horizon of the three paths (and of the cut prediction kept flat); the predicted effect (treated - hold) and whether the
patients with a larger predicted effect have a larger realized difference (truth - hold). Associations, not treatment effects: starts follow flares.

    python -m experiments.latent.eval_treatment_dynamics runs/hs_seq_<stamp> --cpu [--max_events 600]
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.latent import run_hs_seq as R
from experiments.latent.plot_rollout import HORIZON, true_curve
from experiments.latent.plot_rows import BLUE, GRID, INK, MUTED, ORANGE, SURFACE
from experiments.latent.run import build_model

AQUA = "#1baf7a"  # categorical slot 3


def simulate(model, data, z0, Usteps, noise, sub):
    S = z0.shape[0]
    mu = torch.zeros(S, 0)
    dt = (1.0 / R.T_SCALE) / sub
    z, means = z0, []
    for h in range(HORIZON + 1):
        means.append(model.decoder(z, mu).mean[:, 0] * data.ystd + data.ymean)
        if h < HORIZON:
            for s in range(sub):
                u = Usteps[h * sub + s].expand(S, -1)
                z = z + model.drift(z, mu, u) * dt + model.diffusion(z, mu, None) * dt ** 0.5 * noise[h * sub + s]
    return torch.stack(means, 1).numpy()  # (S, HORIZON + 1)


@torch.no_grad()
def event_rollouts(model, data, i, k, S=16, sub=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = data.pack([(int(i), k + 1, None)], torch.zeros(1, 0), "cpu")
    model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
    m, logv, _, _ = model.posterior(batch["t"][:, -1:])
    z0 = m[0, 0] + (0.5 * logv[0, 0]).exp() * torch.randn(S, m.shape[-1], generator=g)
    noise = torch.randn(HORIZON * sub, S, m.shape[-1], generator=g)
    a, b = data.off[i], data.off[i + 1]
    mo, U = data.months[a:b], data.U[a:b]
    q = mo[k] + np.arange(HORIZON * sub) / sub
    paths = {"hold": np.repeat(U[k][None], HORIZON * sub, 0), "treated": np.repeat(U[k + 1][None], HORIZON * sub, 0),
             "realized": U[np.clip(np.searchsorted(mo, q, side="right") - 1, 0, len(mo) - 1)]}
    return {name: simulate(model, data, z0, torch.as_tensor(P, dtype=torch.float32), noise, sub) for name, P in paths.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dim_z", type=int, default=32)
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=[128, 128])
    ap.add_argument("--max_events", type=int, default=10 ** 9)
    ap.add_argument("--patient", default="HS028540")
    ap.add_argument("--cut_month", type=float, default=15.2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    data = R.Data(R.LIGHT, treatment_u=True)
    R.CFG["dim_feat"], R.CFG["dim_mu"], R.CFG["dim_u"] = data.X.shape[1], 0, data.U.shape[1]
    R.CFG["dim_z"], R.CFG["decoder"] = args.dim_z, {**R.CFG["decoder"], "hidden": list(args.dec_hidden)}
    model = build_model(R.CFG, None, 1, None)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location="cpu")["model"])
    model.eval()
    ever = data.names.index("Biologic_ever")
    events = []
    for i in data.test_idx:
        a, b = data.off[i], data.off[i + 1]
        e = np.where(data.M[a:b, ever], data.X[a:b, ever], np.nan)
        events += [(int(i), int(k)) for k in np.flatnonzero(np.diff(e) > 1e-6) if data.ym[a + k]]
    rng = np.random.default_rng(0)
    if len(events) > args.max_events:
        events = [events[j] for j in rng.choice(len(events), args.max_events, replace=False)]
    pid = list(data.pid).index(args.patient)
    k_cut = int(np.argmin(np.abs(data.months[data.off[pid]:data.off[pid + 1]] - args.cut_month)))
    if (pid, k_cut) not in events:
        events.append((pid, k_cut))
    print(f"{len(events)} biologic starts", flush=True)

    names = ["hold", "treated", "realized"]
    centre = {n: [] for n in names}
    truth, keep_id = [], []
    for n_done, (i, k) in enumerate(events):
        a, b = data.off[i], data.off[i + 1]
        t = data.months[a:b]
        own = np.where(data.raw["own_score"][1][a:b], data.raw["own_score"][0][a:b], np.nan)
        out = event_rollouts(model, data, i, k, seed=n_done)
        for n in names:
            centre[n].append(out[n].mean(0))
        truth.append(true_curve(t, own, t[k]))
        keep_id.append((i, k))
        if n_done % 300 == 0:
            print(f"  {n_done}/{len(events)}", flush=True)
    centre = {n: np.array(v) for n, v in centre.items()}
    truth = np.array(truth)

    hs = (0, 3, 6, 9, 12)
    print(f"\nRMSE by horizon over the starts with a defined truth ({'/'.join(str(int(np.isfinite(truth[:, h]).sum())) for h in hs)} at h = {hs}):")
    print(f"{'path':28s}" + "".join(f"{h:>8d}" for h in hs))
    rows = {"flat (cut prediction)": np.repeat(centre["hold"][:, :1], HORIZON + 1, 1), "hold (no change)": centre["hold"],
            "treated (start now)": centre["treated"], "realized (oracle path)": centre["realized"]}
    rmse = {}
    for name, c in rows.items():
        rmse[name] = [np.sqrt(np.nanmean((c[:, h] - truth[:, h]) ** 2)) for h in range(HORIZON + 1)]
        print(f"{name:28s}" + "".join(f"{rmse[name][h]:8.3f}" for h in hs))
    effect = centre["treated"] - centre["hold"]
    resid = truth - centre["hold"]
    print("\npredicted effect (treated - hold), mean over starts [10%, 90%]:")
    for h in (3, 6, 12):
        print(f"  h={h:2d}: {effect[:, h].mean():+.2f}  [{np.percentile(effect[:, h], 10):+.2f}, {np.percentile(effect[:, h], 90):+.2f}]   "
              f"mean realized (truth - hold): {np.nanmean(resid[:, h]):+.2f}")
    print("heterogeneity: do larger predicted effects go with larger realized differences?")
    for h in (3, 6, 12):
        ok = np.isfinite(resid[:, h])
        r = np.corrcoef(effect[ok, h], resid[ok, h])[0, 1]
        slope = np.polyfit(effect[ok, h], resid[ok, h], 1)[0]
        print(f"  h={h:2d}: correlation {r:+.3f}, slope {slope:+.2f}  (n = {ok.sum()})")

    # ---- figure
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 9.5})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    grid = np.arange(HORIZON + 1)
    a0 = ax[0, 0]
    for name, col, ls in (("flat (cut prediction)", MUTED, ":"), ("hold (no change)", BLUE, "-"), ("treated (start now)", ORANGE, "-"),
                          ("realized (oracle path)", AQUA, "-")):
        a0.plot(grid, rmse[name], color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=name)
    a0.set_xlabel("months after the cut")
    a0.set_ylabel("RMSE of the next-6-month mean")
    a0.set_title(f"error by horizon, {len(events)} biologic starts", loc="left", fontsize=10.5)
    a0.legend(frameon=False, fontsize=8.5)
    a1 = ax[0, 1]
    a1.fill_between(grid, np.percentile(effect, 10, 0), np.percentile(effect, 90, 0), color=ORANGE, alpha=0.18, linewidth=0, label="10% to 90% of the starts")
    a1.plot(grid, effect.mean(0), color=ORANGE, lw=2, label="mean predicted effect (treated - hold)")
    a1.plot(grid, np.nanmean(resid, 0), color=INK, lw=1.6, label="mean realized (truth - hold)")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xlabel("months after the cut")
    a1.set_ylabel("change of the next-6-month mean")
    a1.set_title("predicted effect of starting the biologic and the realized difference", loc="left", fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.5)
    a2 = ax[1, 0]
    h = 6
    ok = np.isfinite(resid[:, h])
    edges = np.quantile(effect[ok, h], np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, effect[ok, h], side="right") - 1, 0, 9)
    xs = [effect[ok, h][bi == j].mean() for j in range(10)]
    ys = [resid[ok, h][bi == j].mean() for j in range(10)]
    a2.plot(xs, ys, "o-", color=ORANGE, lw=1.8, ms=5)
    lim = [min(min(xs), min(ys)), max(max(xs), max(ys))]
    a2.plot(lim, lim, color=MUTED, ls=":", lw=0.9, label="realized = predicted")
    a2.set_xlabel("predicted effect at 6 months (mean in the decile)")
    a2.set_ylabel("realized difference (truth - hold)")
    a2.set_title(f"do the predicted differences between patients show up? r = {np.corrcoef(effect[ok, h], resid[ok, h])[0, 1]:+.2f}",
                 loc="left", fontsize=10.5)
    a2.legend(frameon=False, fontsize=8.5)
    a3 = ax[1, 1]
    j = keep_id.index((pid, k_cut))
    t = data.months[data.off[pid]:data.off[pid + 1]]
    own = np.where(data.raw["own_score"][1][data.off[pid]:data.off[pid + 1]], data.raw["own_score"][0][data.off[pid]:data.off[pid + 1]], np.nan)
    past, fut = np.arange(len(t)) <= k_cut, np.arange(len(t)) > k_cut
    a3.plot(t[past], own[past], color=MUTED, lw=0.8, alpha=0.7)
    a3.scatter(t[past], own[past], s=12, color=MUTED, linewidths=0, label="scores up to the cut")
    a3.scatter(t[fut], own[fut], s=22, facecolor="none", edgecolor=MUTED, linewidths=0.9, label="scores after the cut")
    x = t[k_cut] + grid
    a3.plot(x, truth[j], color=INK, lw=2, label="true next-6-month mean")
    for name, col in (("hold", BLUE), ("treated", ORANGE), ("realized", AQUA)):
        a3.plot(x, centre[name][j], color=col, lw=2, label=f"rollout: {name}")
    a3.axvline(t[k_cut], color=MUTED, lw=0.9, ls="--")
    a3.set_xlim(t[k_cut] - 12, t[k_cut] + 18)
    a3.set_xlabel("month")
    a3.set_title(f"{args.patient}, cut at month {t[k_cut]:.1f}", loc="left", fontsize=10.5)
    a3.legend(frameon=False, fontsize=8, ncol=2)
    for a_ in ax.ravel():
        a_.grid(color=GRID, lw=0.7)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    fig.tight_layout()
    out = f"{args.run}/treatment_dynamics.png"
    fig.savefig(out, dpi=105)
    print("saved", out)


if __name__ == "__main__":
    main()
