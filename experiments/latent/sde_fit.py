"""
Route A, step 2: a latent SDE on the frozen GRU states (sde_states.py), to generate trajectories.

State z (32 whitened principal components of the GRU state of a step) between visit steps:
    z(t + dt) ~ N(z + f(z, plan) dt, diag g(z, plan)^2 dt)       (Euler-Maruyama; dt in months; plan = none / 3 drugs)
fitted on transitions of 1, 2, 3 and 5 visits ahead (so that the step size varies and the scheme is consistent in dt), with a constant
plan in the span. A decoder maps z to the score of a visit at that time (Laplace in log1p(score), so scores stay >= 0). Every fifth step adds a rollout term:
the energy score of Euler paths over the span of a transition against the state of the later visit (trains drift and spread of the multi-step law). Generation: start at the state of any step (for
example the decision step of a start), choose a plan, run the SDE on a 7-day grid, decode a score at the days of interest.
The GRU and its heads are not touched, so the forecasts of head 1 and head 2 are unchanged.

Checks on the held-out patients (never used here):
  1. windows: for (step, horizon) pairs as in hs_task2.check_plan_head, the mean score of the visits in the 6-month window, generated at the
     same visit days (24 trajectories), against the observed mean: MAE of the median, share of observed means inside the 5-95% range of the
     trajectories; head 2 and persistence for comparison; also the SDE without noise (drift only);
  2. scores: quantiles and spread of the generated scores against the observed scores in those windows;
  3. drug paths: from the decision step of Adalimumab starts, the generated mean score 1, 3, 6, 9, 12 months after the start with the drug and
     with plan none, against the observed mean score of the same patients (observational: the plan none path is a counterfactual).
Writes <run>/sde.pt, <run>/sde.txt (the report) and <run>/sde_traj.png (example trajectories).

    python -m experiments.latent.sde_fit runs/hs_e2e_<stamp> [--seconds 150]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from experiments.latent.hs_e2e import DECISION, DRUGS, MONTH, plan_at
from experiments.latent.hs_task2 import load

STEP = 7.0  # days per Euler step in generation
N_TRAJ = 24


class LatentSDE(nn.Module):
    def __init__(self, d=32, hidden=256, n_plan=4):
        super().__init__()
        self.n_plan = n_plan
        self.drift = nn.Sequential(nn.Linear(d + n_plan, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))
        self.diff = nn.Sequential(nn.Linear(d + n_plan, 128), nn.SiLU(), nn.Linear(128, d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 2))
        self.register_buffer("lm", torch.zeros(()))  # mean and sd of log1p(score): the decoder works in this space (scores stay >= 0)
        self.register_buffer("ls", torch.ones(()))

    def to_score(self, x):
        return torch.expm1(x * self.ls + self.lm).clamp(min=0.0)

    def fg(self, z, plan):
        u = torch.cat([z, Fn.one_hot(plan, self.n_plan).float()], -1)
        return self.drift(u), Fn.softplus(self.diff(u) - 2.0) + 1e-3

    def decode(self, z):
        o = self.dec(z)
        return o[..., 0], Fn.softplus(o[..., 1]) + 1e-2

    def nll(self, z, plan, dt, z_next):
        f, g = self.fg(z, plan)
        var = g ** 2 * dt[:, None] + 1e-4
        return (0.5 * ((z_next - z - f * dt[:, None]) ** 2 / var + var.log())).sum(1).mean()


def rollout(sde, z0, plan, days, noise=1.0, gen=None, max_steps=40):
    """differentiable Euler-Maruyama paths of z over `days` (n,) with grid steps of about STEP days (the scheme of simulate)"""
    n_steps = torch.clamp(torch.round(days / STEP), 1, max_steps).long()
    h = (days / n_steps.float() / MONTH)[:, None]
    z = z0
    for s_ in range(int(n_steps.max().item())):
        f, g = sde.fg(z, plan)
        z_new = z + f * h + noise * g * h.sqrt() * torch.randn(z.shape, device=z.device, generator=gen)
        z = torch.where((s_ < n_steps)[:, None], z_new.clamp(-12.0, 12.0), z)
    return z


@torch.no_grad()
def simulate(sde, z0, plan, days, n_traj, noise=1.0, gen=None, median=False):
    """z0 (n, d), plan (n,), days (n, m) integer day offsets (>= 0) at which to decode a score. Returns scores (n, n_traj, m), in the
    scores (>= 0)"""
    n, m = days.shape
    dev = z0.device
    z = z0[:, None, :].expand(n, n_traj, -1).reshape(n * n_traj, -1).clone()
    pl = plan[:, None].expand(n, n_traj).reshape(-1)
    idx = torch.clamp(torch.round(days.float() / STEP).long(), min=0)  # grid step of each requested day
    n_steps = int(idx.max().item())
    out = torch.zeros(n, n_traj, m, device=dev)
    dt = STEP / MONTH
    for s in range(n_steps + 1):
        hit = (idx == s)
        if hit.any():
            mu, b = sde.decode(z)
            u = torch.rand(mu.shape, device=dev, generator=gen) - 0.5
            x = mu if median else mu - b * torch.sign(u) * torch.log1p(-2 * u.abs() + 1e-6)  # median: the course of the path, no visit noise
            samp = sde.to_score(x).view(n, n_traj)
            out = torch.where(hit[:, None, :], samp[:, :, None].expand(-1, -1, m), out)
        if s < n_steps:
            f, g = sde.fg(z, pl)
            z = z + f * dt + noise * g * np.sqrt(dt) * torch.randn(z.shape, device=dev, generator=gen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--roll_batch", type=int, default=256, help="pairs per rollout (energy score) term")
    ap.add_argument("--roll_every", type=int, default=5, help="a rollout term every n steps (0: none)")
    ap.add_argument("--roll_weight", type=float, default=1.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, ck = load(args.run, device)
    c = np.load(f"{args.run}/sde_cache.npz")
    Z = torch.as_tensor(c["Z"], device=device)
    P = c["pairs"]
    own = data.names.index("own_score")
    lg = np.log1p(data.score[data.has_score])
    sm, ss = float(lg.mean()), float(lg.std())
    score_n = torch.as_tensor(np.where(data.has_score, (np.log1p(np.maximum(data.score, 0)) - sm) / ss, 0).astype(np.float32), device=device)
    has_score = torch.as_tensor(data.has_score, device=device)
    pid = P[:, 0].astype(np.int64)
    held = data.flag[pid]
    val = ~held & (pid % 20 == 0)
    tr = ~held & ~val
    T = {k: torch.as_tensor(v, device=device) for k, v in dict(a=P[:, 1].astype(np.int64), b=P[:, 2].astype(np.int64), dt=P[:, 3].astype(np.float32),
                                                                plan=P[:, 4].astype(np.int64)).items()}
    tr_idx, val_idx = np.flatnonzero(tr), np.flatnonzero(val)
    sde = LatentSDE().to(device)
    sde.lm.fill_(sm)
    sde.ls.fill_(ss)
    opt = torch.optim.Adam(sde.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    log = open(f"{args.run}/sde.txt", "w")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def batch_loss(ix):
        ix = torch.as_tensor(ix, device=device)
        a, b, dt, pl = T["a"][ix], T["b"][ix], T["dt"][ix], T["plan"][ix]
        l_sde = sde.nll(Z[a], pl, dt, Z[b])
        mu, bb = sde.decode(Z[b])
        ok = has_score[b]
        l_dec = ((score_n[b] - mu).abs() / bb + bb.log())[ok].mean()
        return l_sde, l_dec

    long_idx = tr_idx[P[tr_idx, 3] > 0.4]  # spans of at least 12 days

    def energy_loss(ix, noise=1.0):
        """energy score of the state after a rollout over the span (2 independent paths) against the state of the later visit: a proper
        scoring rule, so it trains both the mean (drift) and the spread (diffusion) of the multi-step law"""
        ix = torch.as_tensor(ix, device=device)
        a, b, dt, pl = T["a"][ix], T["b"][ix], T["dt"][ix], T["plan"][ix]
        days = dt * MONTH
        z1 = rollout(sde, Z[a], pl, days, noise)
        z2 = rollout(sde, Z[a], pl, days, noise)
        nrm = lambda v: (v.pow(2).sum(1) + 1e-6).sqrt()
        return nrm(z1 - Z[b]).mean() - 0.5 * nrm(z1 - z2).mean()

    say(f"{len(tr_idx)} training, {len(val_idx)} validation, {held.sum()} held-out transitions")
    start, step, best, best_state = time.time(), 0, float("inf"), None
    while time.time() - start < args.seconds:
        l_sde, l_dec = batch_loss(rng.choice(tr_idx, args.batch))
        loss = l_sde + l_dec
        if args.roll_every and step % args.roll_every == 0:
            l_es = energy_loss(rng.choice(long_idx, args.roll_batch))
            loss = loss + args.roll_weight * l_es
        opt.zero_grad()
        loss.backward()
        if torch.isfinite(nn.utils.clip_grad_norm_(sde.parameters(), 10.0)):
            opt.step()
        step += 1
        if step % 500 == 0:
            with torch.no_grad():
                vs, vd = batch_loss(rng.choice(val_idx, 20000))
                ve = energy_loss(rng.choice(val_idx[P[val_idx, 3] > 0.4], 2000))
            say(f"{step} t={time.time() - start:.0f}s train SDE NLL {l_sde.item():.3f} decoder {l_dec.item():.3f} | validation {vs.item():.3f} {vd.item():.3f} energy {ve.item():.3f}")
            if (vs + vd + ve).item() < best:
                best, best_state = (vs + vd + ve).item(), {k: v.clone() for k, v in sde.state_dict().items()}
    sde.load_state_dict(best_state)
    sde.eval()
    torch.save(sde.state_dict(), f"{args.run}/sde.pt")
    say(f"trained {step} steps; best validation loss {best:.3f}")
    evaluate(sde, model, data, c, Z, score_n, sm, ss, device, args.run, say)


@torch.no_grad()
def evaluate(sde, model, data, c, Z, score_n, sm, ss, device, run, say):
    rng = np.random.default_rng(1)
    gen = torch.Generator(device=device).manual_seed(0)
    held = [int(i) for i in np.flatnonzero(data.flag) if data.n_rows(i) > 3]
    j_last = data.names.index("last_score")
    ys, ym = data.stats["ystd"], data.stats["ymean"]
    # ---- 1 and 2: windows
    ex = []
    for i in rng.permutation(held)[:1500]:
        ex += [(i,) + e for e in data.plan_examples(i, rng, 2, partial=False)]
    ex = ex[:3000]
    rows, gens, gen_pool = [], [], []
    for a in range(0, len(ex), 300):
        part = ex[a:a + 300]
        pats = sorted({e[0] for e in part})
        h = model.encode(data.pack(pats, device))
        pos = {p: k for k, p in enumerate(pats)}
        bi = torch.as_tensor([pos[e[0]] for e in part], device=device)
        ki = torch.as_tensor([e[1] for e in part], device=device)
        f = lambda v, t=torch.float32: torch.as_tensor(np.asarray(v), dtype=t, device=device)
        mu2 = (plan_at(model, h, bi, ki, f([e[2] for e in part]), f([e[3] for e in part]), f([e[4] for e in part], torch.long))[0] * ys + ym).cpu().numpy()
        # visit days in each window, relative to the step
        days = np.zeros((len(part), 12), np.int64)
        valid = np.zeros((len(part), 12), bool)
        for r, e in enumerate(part):
            i, kk = e[0], e[1]
            s0, s1 = data.off[i], data.off[i + 1]
            t = data.day[s0 + kk]
            sd = data.day[s0:s1][data.has_score[s0:s1]] - t
            sel = sd[(sd > e[2] * MONTH) & (sd <= (e[2] + 6) * MONTH)][:12]
            days[r, :len(sel)], valid[r, :len(sel)] = sel, True
        z0 = Z[torch.as_tensor([data.off[e[0]] + e[1] for e in part], device=device)]
        plan = f([e[4] for e in part], torch.long)
        dd = torch.as_tensor(days, device=device)
        for noise, store in ((1.0, gens), (0.0, None)):
            sc = simulate(sde, z0, plan, dd, N_TRAJ, noise=noise, gen=gen)   # (n, N, 12)
            vm = torch.as_tensor(valid, device=device)[:, None, :].float()
            wm = ((sc * vm).sum(2) / vm.sum(2)).cpu().numpy()  # (n, N) window mean of the generated scores
            if store is not None:
                gens.append(wm)
                for r in range(len(part)):
                    if valid[r].any():
                        gen_pool.append(sc[r, :, :valid[r].sum()].cpu().numpy().ravel())
            else:
                det = wm.mean(1)
        g_rows = [data.off[e[0]] + e[1] for e in part]
        last = np.where(data.M[g_rows, j_last], data.raw[g_rows, j_last], ym)
        rows.append(np.column_stack([[e[2] for e in part], [e[4] for e in part], [e[5] for e in part], mu2, last, det]))
    R, G = np.concatenate(rows), np.concatenate(gens)
    tgt, med = R[:, 2], np.median(G, 1)
    lo, hi = np.percentile(G, 5, axis=1), np.percentile(G, 95, axis=1)
    say(f"\n1. windows of the held-out patients: {len(R)} (step, horizon) pairs; the mean score of the visits in the 6-month window")
    say(f"   {'horizon (months)':>17s} {'n':>5s} {'SDE median':>11s} {'SDE no noise':>13s} {'head 2':>7s} {'persistence':>12s} {'inside 5-95%':>13s}")
    for lo_h, hi_h in ((0, 6), (6, 12), (12, 36), (36, 120), (0, 120)):
        s = (R[:, 0] >= lo_h) & (R[:, 0] < hi_h)
        if s.sum() < 30:
            continue
        say(f"   {f'{lo_h}-{hi_h}':>17s} {s.sum():5d} {np.abs(med[s] - tgt[s]).mean():11.3f} {np.abs(R[s, 5] - tgt[s]).mean():13.3f} "
            f"{np.abs(R[s, 3] - tgt[s]).mean():7.3f} {np.abs(R[s, 4] - tgt[s]).mean():12.3f} {np.mean((tgt[s] >= lo[s]) & (tgt[s] <= hi[s])):13.3f}")
    # observed scores in the same windows
    ob = []
    for e in ex:
        i, kk = e[0], e[1]
        s0, s1 = data.off[i], data.off[i + 1]
        sd = data.day[s0:s1][data.has_score[s0:s1]] - data.day[s0 + kk]
        sv = data.score[s0:s1][data.has_score[s0:s1]]
        ob.append(sv[(sd > e[2] * MONTH) & (sd <= (e[2] + 6) * MONTH)][:12])
    ob_all = np.concatenate(ob)
    gen_all = np.concatenate([x for x in gen_pool])
    say("\n2. scores in those windows: observed against generated (pooled over windows)")
    say(f"   {'':10s} {'mean':>6s} {'sd':>6s} {'10%':>6s} {'50%':>6s} {'90%':>6s}")
    for nm, v in (("observed", ob_all), ("generated", gen_all)):
        say(f"   {nm:10s} {v.mean():6.2f} {v.std():6.2f} {np.percentile(v, 10):6.2f} {np.percentile(v, 50):6.2f} {np.percentile(v, 90):6.2f}")
    # ---- 3: drug paths from the decision step of held-out Adalimumab starts
    starts = []
    for i in held:
        s0, s1 = data.off[i], data.off[i + 1]
        st, en, dr = data.bio[i]
        hs = data.has_score[s0:s1]
        sd, sv = data.day[s0:s1][hs], data.score[s0:s1][hs]
        for kd in np.flatnonzero(data.kind[s0:s1] == DECISION):
            s = data.day[s0 + kd] + 1
            j = np.flatnonzero((st == s) & (dr == 0))
            if len(j):
                starts.append((s0 + kd, s, sd - s, sv))
    months = (1, 3, 6, 9, 12)
    z0 = Z[torch.as_tensor([x[0] for x in starts], device=device)]
    dd = torch.as_tensor(np.tile([int(m * MONTH) + 1 for m in months], (len(starts), 1)), device=device)
    paths = {}
    for nm, pl in (("Adalimumab", 1), ("none", 0)):
        paths[nm] = simulate(sde, z0, torch.full((len(starts),), pl, device=device), dd, N_TRAJ, gen=gen).mean(1).cpu().numpy()
    say(f"\n3. paths from the decision step of {len(starts)} held-out Adalimumab starts: mean score, observed vs generated "
        f"(observed: visits within 1.5 months of the time, same patients)")
    say(f"   {'months after start':>18s} {'n with a visit':>15s} {'observed':>9s} {'gen. Adalimumab':>16s} {'gen. none':>10s}")
    pre = np.array([x[3][(x[2] >= -180) & (x[2] < 0)].mean() if ((x[2] >= -180) & (x[2] < 0)).any() else np.nan for x in starts])
    say(f"   {'-6..0 (before)':>18s} {np.isfinite(pre).sum():15d} {np.nanmean(pre):9.2f}")
    for k, m in enumerate(months):
        ok = np.array([((x[2] >= (m - 1.5) * MONTH) & (x[2] < (m + 1.5) * MONTH)).any() for x in starts])
        obs_m = np.array([x[3][(x[2] >= (m - 1.5) * MONTH) & (x[2] < (m + 1.5) * MONTH)].mean() if o else np.nan for x, o in zip(starts, ok)])
        say(f"   {m:18d} {ok.sum():15d} {np.nanmean(obs_m):9.2f} {paths['Adalimumab'][ok, k].mean():16.2f} {paths['none'][ok, k].mean():10.2f}")
    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.latent.hs_task2 import open_lines
    lines = [l for l in open_lines(data, held) if data.n_rows(l[0]) > 12][:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=False)
    cut = data.end_day
    for ax, l in zip(axes.ravel(), lines):
        i = l[0]
        s0, s1 = data.off[i], data.off[i + 1]
        hs = data.has_score[s0:s1]
        sd, sv = (data.day[s0:s1][hs] - cut) / MONTH, data.score[s0:s1][hs]
        z0 = Z[torch.as_tensor([s0 + l[4]], device=device)]
        horizon = 12
        grid = np.arange(0, horizon * MONTH, 14)
        dd = torch.as_tensor(grid[None, :], device=device)
        sc = simulate(sde, z0, torch.as_tensor([l[2] + 1], device=device), dd, 200, gen=gen)[0].cpu().numpy()
        x = grid / MONTH
        ax.fill_between(x, np.percentile(sc, 5, 0), np.percentile(sc, 95, 0), color="#7aa6d6", alpha=0.35, lw=0)
        ax.fill_between(x, np.percentile(sc, 25, 0), np.percentile(sc, 75, 0), color="#3b78b8", alpha=0.45, lw=0)
        ax.plot(x, np.median(sc, 0), color="#1f4e86", lw=2)
        ax.scatter(sd[sd > -24], sv[sd > -24], s=14, color="#333333", zorder=3)
        ax.axvline(0, color="#999999", lw=1, ls="--")
        ax.set_title(f"patient {data.pid[i]}, on {DRUGS[l[2]]}", fontsize=10, loc="left")
        ax.set_xlabel("months from the cutoff")
        ax.set_ylabel("IHS4")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].text(0.98, 0.95, "dots: observed visits\nband: generated, 50% and 90%\nline: median", transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(f"{run}/sde_traj.png", dpi=130)
    say(f"\nfigure: {run}/sde_traj.png")


if __name__ == "__main__":
    main()
