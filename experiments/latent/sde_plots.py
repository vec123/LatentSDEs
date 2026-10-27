"""
Plots of the latent SDE (sde_fit.py) and head 2 against the observed course of held-out patients (never used in training). Every rollout starts at a
past step, so the visits after it are the ground truth. Writes <run>/sde_plots/:
  A_patients.png     6 held-out patients (drawn at random, not chosen): history, the visits of the next 12 months (ground truth), the SDE band
                     (50% / 90%, visit scores with visit noise) and median, 4 single rollouts (the decoded course of one latent path, without the
                     noise of a single visit), the head 2 forecast of the two 6-month windows and the observed window means;
  B_calibration.png  6-month window means (step, horizon pairs as in hs_task2.check_plan_head): predicted against observed in tenths of the
                     prediction, for the SDE median and head 2; and the coverage of the central intervals of the SDE trajectories;
  C_drug_paths.png   from the decision step of held-out starts of each drug: the observed mean score by month after the start (95% interval) against
                     the generated mean with the drug, and the generated mean without treatment (a counterfactual: no ground truth).

    python -m experiments.latent.sde_plots runs/hs_e2e_<stamp> [--sde <weights> --out <folder>]        # about 1 minute on the GPU
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.latent.hs_e2e import DECISION, DRUGS, MONTH, VISIT, plan_at
from experiments.latent.hs_task2 import load
from experiments.latent.sde_fit import LatentSDE, simulate
from experiments.latent.sde_states import plan_of_span

C_SDE, C_SDE_L, C_H2, C_TRUTH, C_HIST, C_NONE, C_PATH = "#2b6cb0", "#9cc0e6", "#c05621", "#111111", "#a0a0a0", "#6b6b6b", "#6b46c1"
N_PATHS = 4


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


@torch.no_grad()
def head2(model, data, i, kk, hz, plan, device):
    """head 2 (visit-label scale) at local step kk of patient i: list over horizons hz (months)"""
    h = model.encode(data.pack([i], device))
    st, en, dr = data.bio[i]
    t = data.day[data.off[i] + kk]
    out = []
    for z in hz:
        ws = t + z * MONTH
        act = (st <= ws) & (en >= ws)
        tau = (ws - st[act].max()) / MONTH if plan > 0 and act.any() else 0.0
        f = lambda v, tp=torch.float32: torch.as_tensor([v], dtype=tp, device=device)
        mu = plan_at(model, h, f(0, torch.long), f(kk, torch.long), f(z), f(tau), f(plan, torch.long))[0]
        out.append(float(mu * data.stats["ystd"] + data.stats["ymean"]))
    return out


def fig_patients(sde, model, data, Z, device, gen, out):
    rng = np.random.default_rng(3)
    held = [int(i) for i in np.flatnonzero(data.flag) if data.n_rows(i) > 20]
    picks = []
    for i in rng.permutation(held):
        s0, s1 = data.off[i], data.off[i + 1]
        vr = np.flatnonzero((data.kind[s0:s1] == VISIT) & data.has_score[s0:s1])
        days = data.day[s0 + vr]
        ok = [k for k, t in zip(vr, days) if t <= data.end_day - 365 and ((days > t) & (days <= t + 365)).sum() >= 4 and (days < t).sum() >= 4]
        if not ok:
            continue
        kk = int(ok[-1])
        t0 = data.day[s0 + kk]
        st, en, dr = data.bio[i]
        plan = plan_of_span(st, en, dr, t0, t0 + 365)
        if plan < 0:
            continue
        picks.append((i, kk, plan))
        if len(picks) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    grid = np.arange(0, 366, 7)
    for n_, (ax, (i, kk, plan)) in enumerate(zip(axes.ravel(), picks)):
        s0, s1 = data.off[i], data.off[i + 1]
        t0 = data.day[s0 + kk]
        hs = data.has_score[s0:s1]
        sd, sv = (data.day[s0:s1][hs] - t0) / MONTH, data.score[s0:s1][hs]
        sc = simulate(sde, Z[[s0 + kk]], torch.as_tensor([plan], device=device), torch.as_tensor(grid[None, :], device=device), 400, gen=gen)[0].cpu().numpy()
        x = grid / MONTH
        ax.fill_between(x, np.percentile(sc, 5, 0), np.percentile(sc, 95, 0), color=C_SDE_L, alpha=0.55, lw=0, label="SDE 90% band")
        ax.fill_between(x, np.percentile(sc, 25, 0), np.percentile(sc, 75, 0), color=C_SDE, alpha=0.35, lw=0, label="SDE 50% band")
        ax.plot(x, np.median(sc, 0), color=C_SDE, lw=2, label="SDE median")
        paths = simulate(sde, Z[[s0 + kk]], torch.as_tensor([plan], device=device), torch.as_tensor(grid[None, :], device=device), N_PATHS, gen=gen,
                         median=True)[0].cpu().numpy()
        for q, pth in enumerate(paths):
            ax.plot(x, pth, color=C_PATH, lw=0.9, alpha=0.85, label=f"{N_PATHS} single rollouts (course, no visit noise)" if q == 0 else None)
        past, fut = (sd <= 0) & (sd > -24), (sd > 0) & (sd <= 12)
        ax.scatter(sd[past], sv[past], s=16, color=C_HIST, zorder=3, label="history (seen by the model)")
        ax.scatter(sd[fut], sv[fut], s=26, color=C_TRUTH, edgecolor="white", lw=0.8, zorder=4, label="ground truth (not seen)")
        h2 = head2(model, data, i, kk, [0, 6], plan, device)
        for w, (a, b) in enumerate(((0, 6), (6, 12))):
            ax.plot([a, b], [h2[w]] * 2, color=C_H2, lw=2.5, solid_capstyle="butt", label="head 2: 6-month mean" if w == 0 else None)
            obs = sv[(sd > a) & (sd <= b)]
            if len(obs):
                ax.plot([a, b], [obs.mean()] * 2, color=C_TRUTH, lw=1.5, ls="--", label="observed 6-month mean" if w == 0 else None)
        ax.axvline(0, color="#999999", lw=1, ls=":")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{data.pid[i]}, plan: {(['no biologic'] + DRUGS)[plan]}", loc="left", fontsize=10)
        ax.set_xlabel("months from the start of the rollout")
        if n_ % 3 == 0:
            ax.set_ylabel("IHS4")
        style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Held-out patients: rollout from a past visit against the next 12 months (random patients)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


@torch.no_grad()
def fig_calibration(sde, model, data, Z, device, gen, out):
    rng = np.random.default_rng(1)
    held = [int(i) for i in np.flatnonzero(data.flag) if data.n_rows(i) > 3]
    ex = []
    for i in rng.permutation(held)[:2500]:
        ex += [(i,) + e for e in data.plan_examples(i, rng, 2, partial=False)]
    ys, ym = data.stats["ystd"], data.stats["ymean"]
    T, H2, G = [], [], []
    for a in range(0, len(ex), 300):
        part = ex[a:a + 300]
        pats = sorted({e[0] for e in part})
        h = model.encode(data.pack(pats, device))
        pos = {p: k for k, p in enumerate(pats)}
        f = lambda v, t=torch.float32: torch.as_tensor(np.asarray(v), dtype=t, device=device)
        H2.append((plan_at(model, h, f([pos[e[0]] for e in part], torch.long), f([e[1] for e in part], torch.long), f([e[2] for e in part]),
                           f([e[3] for e in part]), f([e[4] for e in part], torch.long))[0] * ys + ym).cpu().numpy())
        days, valid = np.zeros((len(part), 12), np.int64), np.zeros((len(part), 12), bool)
        for r, e in enumerate(part):
            s0, s1 = data.off[e[0]], data.off[e[0] + 1]
            sd = data.day[s0:s1][data.has_score[s0:s1]] - data.day[s0 + e[1]]
            sel = sd[(sd > e[2] * MONTH) & (sd <= (e[2] + 6) * MONTH)][:12]
            days[r, :len(sel)], valid[r, :len(sel)] = sel, True
        sc = simulate(sde, Z[[data.off[e[0]] + e[1] for e in part]], f([e[4] for e in part], torch.long), torch.as_tensor(days, device=device), 64, gen=gen)
        vm = torch.as_tensor(valid, device=device)[:, None, :].float()
        G.append(((sc * vm).sum(2) / vm.sum(2)).cpu().numpy())
        T += [e[5] for e in part]
    T, H2, G = np.array(T), np.concatenate(H2), np.concatenate(G)
    med = np.median(G, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for pred, nm, c in ((med, "SDE median", C_SDE), (H2, "head 2", C_H2)):
        q = np.quantile(pred, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, 9)
        px, py = [pred[b == k].mean() for k in range(10)], [T[b == k].mean() for k in range(10)]
        a1.plot(px, py, "-o", color=c, lw=2, ms=6, label=f"{nm} (MAE {np.abs(pred - T).mean():.2f})")
    lim = [0, max(np.max(px), np.max(py)) * 1.08]
    a1.plot(lim, lim, color="#999999", lw=1, ls="--", label="perfect")
    a1.set_xlabel("predicted 6-month mean (tenths of the prediction)")
    a1.set_ylabel("observed 6-month mean")
    a1.set_title(f"Window means of held-out patients ({len(T)} step, horizon pairs)", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=9)
    style(a1)
    lv = np.linspace(0.1, 0.9, 9)
    cov = [np.mean((T >= np.percentile(G, 50 - 50 * l, axis=1)) & (T <= np.percentile(G, 50 + 50 * l, axis=1))) for l in lv]
    a2.plot(lv, cov, "-o", color=C_SDE, lw=2, ms=6, label="SDE trajectories")
    a2.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect")
    a2.set_xlabel("nominal level of the central interval")
    a2.set_ylabel("share of observed means inside")
    a2.set_title("Coverage of the SDE intervals", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=9)
    style(a2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return len(T), np.abs(med - T).mean(), np.abs(H2 - T).mean(), dict(zip(np.round(lv, 1), np.round(cov, 3)))


@torch.no_grad()
def fig_drug_paths(sde, data, Z, device, gen, out):
    held = [int(i) for i in np.flatnonzero(data.flag)]
    months = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for d_, ax in enumerate(axes):
        starts = []
        for i in held:
            s0, s1 = data.off[i], data.off[i + 1]
            st, en, dr = data.bio[i]
            hs = data.has_score[s0:s1]
            sd, sv = data.day[s0:s1][hs], data.score[s0:s1][hs]
            for kd in np.flatnonzero(data.kind[s0:s1] == DECISION):
                s = data.day[s0 + kd] + 1
                if ((st == s) & (dr == d_)).any():
                    starts.append((s0 + kd, (sd - s) / MONTH, sv))
        z0 = Z[torch.as_tensor([x[0] for x in starts], device=device)]
        dd = torch.as_tensor(np.tile((months * MONTH).astype(np.int64), (len(starts), 1)), device=device)
        gd = simulate(sde, z0, torch.full((len(starts),), d_ + 1, device=device), dd, 16, gen=gen).mean(1).cpu().numpy()
        g0 = simulate(sde, z0, torch.zeros(len(starts), dtype=torch.long, device=device), dd, 16, gen=gen).mean(1).cpu().numpy()
        om, olo, ohi, gm, gn = [], [], [], [], []
        for k, m in enumerate(months):
            vals, idx = [], []
            for j, x in enumerate(starts):
                v = x[2][(x[1] > m - 0.5) & (x[1] <= m + 0.5)]
                if len(v):
                    vals.append(v.mean()); idx.append(j)
            vals = np.array(vals)
            om.append(vals.mean()); se = vals.std() / np.sqrt(len(vals))
            olo.append(vals.mean() - 1.96 * se); ohi.append(vals.mean() + 1.96 * se)
            gm.append(gd[idx, k].mean()); gn.append(g0[idx, k].mean())
        pre = [x[2][(x[1] > -6) & (x[1] <= 0)].mean() for x in starts if ((x[1] > -6) & (x[1] <= 0)).any()]
        ax.fill_between(months, olo, ohi, color="#cfcfcf", lw=0, label="observed, 95% interval")
        ax.plot(months, om, "-o", color=C_TRUTH, lw=2, ms=5, label="observed (ground truth)")
        ax.plot(months, gm, "-o", color=C_SDE, lw=2, ms=5, label="generated, with the drug")
        ax.plot(months, gn, "--", color=C_NONE, lw=2, label="generated, no treatment (counterfactual)")
        ax.scatter([0], [np.mean(pre)], color=C_TRUTH, marker="s", s=40, zorder=4, label="observed mean, 6 months before")
        ax.set_title(f"{DRUGS[d_]}: {len(starts)} held-out starts", loc="left", fontsize=10)
        ax.set_xlabel("months after the start")
        style(ax)
    axes[0].set_ylabel("mean IHS4")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Course after a biologic start: observed against generated from the decision step (the observed course includes later stops)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)


@torch.no_grad()
def pitch_picks(sde, data, Z, device, gen, scan=2500):
    """Selected held-out examples for a presentation, one per plan group (no biologic, Adalimumab, Bimekizumab / Secukinumab). Chosen, not random:
    every visit of the next 12 months inside the 90% band, the mean score of those visits >= 3 points away from the last 6 months (a real change),
    then the smallest relative error of the median. Where shown, say that they are selected and show B_calibration (all patients) beside them.
    Returns the picks and all scored candidates, each (relative error, inside share, change, (patient, local step, plan, future days, future
    scores))."""
    rng = np.random.default_rng(5)
    held = [int(i) for i in np.flatnonzero(data.flag) if data.n_rows(i) > 15]
    cands = []
    for i in rng.permutation(held)[:scan]:
        s0, s1 = data.off[i], data.off[i + 1]
        vr = np.flatnonzero((data.kind[s0:s1] == VISIT) & data.has_score[s0:s1])
        days = data.day[s0 + vr]
        st, en, dr = data.bio[i]
        for k, t in zip(vr[::-1], days[::-1]):
            if t > data.end_day - 365:
                continue
            fut = (days > t) & (days <= t + 365)
            if fut.sum() >= 5 and (days < t).sum() >= 5:
                p = plan_of_span(st, en, dr, t, t + 365)
                if p >= 0:
                    cands.append((i, int(k), p, days[fut] - t, data.score[s0 + vr][fut]))
            break
    res = []
    for a in range(0, len(cands), 300):
        part = cands[a:a + 300]
        dd, valid = np.zeros((len(part), 12), np.int64), np.zeros((len(part), 12), bool)
        for r, c in enumerate(part):
            n = min(len(c[3]), 12)
            dd[r, :n], valid[r, :n] = c[3][:n], True
        sc = simulate(sde, Z[[data.off[c[0]] + c[1] for c in part]], torch.as_tensor([c[2] for c in part], device=device),
                      torch.as_tensor(dd, device=device), 64, gen=gen).cpu().numpy()
        for r, c in enumerate(part):
            n = int(valid[r].sum())
            v = sc[r, :, :n]
            obs = c[4][:n]
            inside = np.mean((obs >= np.percentile(v, 5, 0)) & (obs <= np.percentile(v, 95, 0)))
            s0, s1 = data.off[c[0]], data.off[c[0] + 1]
            hs = data.has_score[s0:s1]
            sd = data.day[s0:s1][hs] - data.day[s0 + c[1]]
            past = data.score[s0:s1][hs][(sd <= 0) & (sd > -183)]
            change = abs(obs.mean() - past.mean()) if len(past) else 0.0
            res.append((np.abs(np.median(v, 0) - obs).mean() / max(obs.mean(), 1.0), inside, change, c))
    picks = []
    for grp in ((0,), (1,), (2, 3)):
        g = [x for x in res if x[3][2] in grp and x[1] == 1.0 and x[2] >= 3] or [x for x in res if x[3][2] in grp and x[1] == 1.0]
        if g:
            picks.append(min(g, key=lambda x: x[0]))
    print(f"P: {len(cands)} candidates scanned; picks: " + "; ".join(f"{data.pid[x[3][0]]} plan {x[3][2]} rel. error {x[0]:.2f} change {x[2]:.1f}" for x in picks))
    return picks, res


def fig_pitch(sde, model, data, Z, device, gen, out, scan=2500):
    """The examples of pitch_picks: history, the SDE band and median, 4 single rollouts and the visits of the next 12 months."""
    picks, _ = pitch_picks(sde, data, Z, device, gen, scan)
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 5.6), facecolor="#FCFBF8")
    grid = np.arange(0, 366, 7)
    x = grid / MONTH
    for n_, (ax, (_, _, _, (i, kk, plan, _, _))) in enumerate(zip(np.atleast_1d(axes), picks)):
        ax.set_facecolor("#FCFBF8")
        s0, s1 = data.off[i], data.off[i + 1]
        t0 = data.day[s0 + kk]
        hs = data.has_score[s0:s1]
        sd, sv = (data.day[s0:s1][hs] - t0) / MONTH, data.score[s0:s1][hs]
        z0, pl, dd = Z[[s0 + kk]], torch.as_tensor([plan], device=device), torch.as_tensor(grid[None, :], device=device)
        sc = simulate(sde, z0, pl, dd, 400, gen=gen)[0].cpu().numpy()
        ax.fill_between(x, np.percentile(sc, 5, 0), np.percentile(sc, 95, 0), color=C_SDE_L, alpha=0.6, lw=0, label="90% interval")
        ax.fill_between(x, np.percentile(sc, 25, 0), np.percentile(sc, 75, 0), color=C_SDE, alpha=0.35, lw=0, label="50% interval")
        for q, pth in enumerate(simulate(sde, z0, pl, dd, N_PATHS, gen=gen, median=True)[0].cpu().numpy()):
            ax.plot(x, pth, color=C_PATH, lw=1.0, alpha=0.8, label="generated trajectories" if q == 0 else None)
        ax.plot(x, np.median(sc, 0), color=C_SDE, lw=2.5, label="predicted (median)")
        past, fut = (sd <= 0) & (sd > -18), (sd > 0) & (sd <= 12)
        ax.scatter(sd[past], sv[past], s=30, color=C_HIST, zorder=3, label="history (model input)")
        ax.scatter(sd[fut], sv[fut], s=48, color=C_TRUTH, edgecolor="white", lw=1, zorder=4, label="truth (not seen)")
        ax.axvline(0, color="#999999", lw=1, ls=":")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-18, 12.5)
        ax.set_title(f"Patient {'ABC'[n_]}: {(['no biologic', 'on Adalimumab', 'on Bimekizumab', 'on Secukinumab'])[plan]}", loc="left", fontsize=17)
        ax.set_xlabel("months from the forecast date", fontsize=15)
        if n_ == 0:
            ax.set_ylabel("IHS4 score", fontsize=15)
        style(ax)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


@torch.no_grad()
def fig_counterfactual(sde, model, data, Z, device, gen, out, scan=4203):
    """Selected held-out biologic starts, one per drug: from the decision step (the day before the start) the SDE course with the drug the patient
    got and without any biologic (the counterfactual: no ground truth), the observed visits after the start (under the drug), and the plan head's
    6-month means for both plans (the plot shows 50% bands). Chosen, not random: every visit of the next 12 months inside the 90% band of the drug course, then the largest
    predicted difference between the two plans. Where shown, say that they are selected."""
    cands = []
    for i in np.flatnonzero(data.flag)[:scan]:
        s0, s1 = data.off[i], data.off[i + 1]
        st, en, dr = data.bio[i]
        hs = data.has_score[s0:s1]
        sd, sv = data.day[s0:s1][hs], data.score[s0:s1][hs]
        for kd in np.flatnonzero(data.kind[s0:s1] == DECISION):
            s = data.day[s0 + kd] + 1
            j = np.flatnonzero(st == s)
            if not len(j):
                continue
            j = int(j[0])
            others = np.arange(len(st)) != j
            if en[j] < s + 365 or (others & (st <= s + 365) & (en >= s)).any():  # the drug alone through the 12 months
                continue
            fut = (sd > s) & (sd <= s + 365)
            if fut.sum() >= 4 and ((sd >= s - 365) & (sd < s)).sum() >= 2:
                cands.append((int(i), int(kd), int(dr[j]), sd[fut] - (s - 1), sv[fut]))
    res = []
    for a in range(0, len(cands), 300):
        part = cands[a:a + 300]
        dd = np.zeros((len(part), 12), np.int64)
        valid = np.zeros((len(part), 12), bool)
        for r, c in enumerate(part):
            n = min(len(c[3]), 12)
            dd[r, :n], valid[r, :n] = c[3][:n], True
        z0 = Z[[data.off[c[0]] + c[1] for c in part]]
        ddt = torch.as_tensor(dd, device=device)
        sc_d = simulate(sde, z0, torch.as_tensor([c[2] + 1 for c in part], device=device), ddt, 64, gen=gen).cpu().numpy()
        sc_0 = simulate(sde, z0, torch.zeros(len(part), dtype=torch.long, device=device), ddt, 64, gen=gen).cpu().numpy()
        for r, c in enumerate(part):
            n = int(valid[r].sum())
            obs = c[4][:n]
            inside = np.mean((obs >= np.percentile(sc_d[r, :, :n], 5, 0)) & (obs <= np.percentile(sc_d[r, :, :n], 95, 0)))
            rel = np.abs(np.median(sc_d[r, :, :n], 0) - obs).mean() / max(obs.mean(), 1.0)
            effect = np.median(sc_0[r, :, :n], 0).mean() - np.median(sc_d[r, :, :n], 0).mean()
            res.append((inside, rel, effect, c))
    picks = []
    for d_ in range(len(DRUGS)):
        g = [x for x in res if x[3][2] == d_ and x[0] == 1.0 and x[1] <= 0.25]
        if g:
            picks.append(max(g, key=lambda x: x[2]))
    print(f"CF: {len(cands)} held-out starts with 12 months on the drug; picks: " +
          "; ".join(f"{data.pid[x[3][0]]} {DRUGS[x[3][2]]} effect {x[2]:.1f} rel. error {x[1]:.2f}" for x in picks))
    plt.rcParams.update({"font.size": 14})
    C_CF, C_CF_L = "#C05621", "#F3B98F"
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 5.6), facecolor="#FCFBF8")
    grid = np.arange(0, 367, 7)
    x = grid / MONTH
    for n_, (ax, (_, _, _, (i, kd, d_, _, _))) in enumerate(zip(np.atleast_1d(axes), picks)):
        ax.set_facecolor("#FCFBF8")
        s0, s1 = data.off[i], data.off[i + 1]
        t0 = data.day[s0 + kd]
        hs = data.has_score[s0:s1]
        sd, sv = (data.day[s0:s1][hs] - t0) / MONTH, data.score[s0:s1][hs]
        z0, dd = Z[[s0 + kd]], torch.as_tensor(grid[None, :], device=device)
        sc_d = simulate(sde, z0, torch.as_tensor([d_ + 1], device=device), dd, 400, gen=gen)[0].cpu().numpy()
        sc_0 = simulate(sde, z0, torch.zeros(1, dtype=torch.long, device=device), dd, 400, gen=gen)[0].cpu().numpy()
        ax.fill_between(x, np.percentile(sc_0, 25, 0), np.percentile(sc_0, 75, 0), color=C_CF_L, alpha=0.5, lw=0, label="no biologic: 50% interval")
        ax.fill_between(x, np.percentile(sc_d, 25, 0), np.percentile(sc_d, 75, 0), color=C_SDE_L, alpha=0.65, lw=0, label="with the drug: 50% interval")
        ax.plot(x, np.median(sc_0, 0), color=C_CF, lw=2.5, ls="--", label="no biologic (counterfactual)")
        ax.plot(x, np.median(sc_d, 0), color=C_SDE, lw=2.5, label="with the drug (predicted)")
        past, fut = (sd <= 0) & (sd > -12), (sd > 0) & (sd <= 12)
        ax.scatter(sd[past], sv[past], s=30, color=C_HIST, zorder=3, label="history (model input)")
        ax.scatter(sd[fut], sv[fut], s=48, color=C_TRUTH, edgecolor="white", lw=1, zorder=4, label="truth under the drug")
        h_d = head2(model, data, i, kd, [0], d_ + 1, device)[0]
        h_0 = head2(model, data, i, kd, [0], 0, device)[0]
        ax.text(0.98, 0.97, f"plan head, months 0-6:\n{h_d:.1f} with {DRUGS[d_]}\n{h_0:.1f} without", transform=ax.transAxes, ha="right", va="top", fontsize=13,
                color="#333333", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D8D4CA"))
        ax.axvline(0, color="#999999", lw=1, ls=":")
        top = max(sv[past | fut].max(), np.percentile(sc_0, 75, 0).max(), np.percentile(sc_d, 75, 0).max())
        ax.set_ylim(0, top * 1.45)  # room for the plan-head box
        ax.set_xlim(-12, 12.3)
        ax.set_title(f"Patient {'DEF'[n_]}: starts {DRUGS[d_]}", loc="left", fontsize=17)
        ax.set_xlabel("months from the biologic start", fontsize=15)
        if n_ == 0:
            ax.set_ylabel("IHS4 score", fontsize=15)
        style(ax)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_simple(sde, data, Z, device, gen, out, pid="HS030725", bg="#111827"):
    """One selected held-out start (patient D of F_counterfactual_examples.png) for a non-technical slide: the course with the drug and without any
    biologic (median and 50% band), the visits before the start and the visits under the drug; plain labels, dark background."""
    i = int(np.flatnonzero(data.pid == pid)[0])
    s0, s1 = data.off[i], data.off[i + 1]
    st, en, dr = data.bio[i]
    kd = next(int(k) for k in np.flatnonzero(data.kind[s0:s1] == DECISION) if (st == data.day[s0 + k] + 1).any())
    d_ = int(dr[np.flatnonzero(st == data.day[s0 + kd] + 1)[0]])
    t0 = data.day[s0 + kd]
    hs = data.has_score[s0:s1]
    sd, sv = (data.day[s0:s1][hs] - t0) / MONTH, data.score[s0:s1][hs]
    grid = np.arange(0, 367, 7)
    x = grid / MONTH
    z0, dd = Z[[s0 + kd]], torch.as_tensor(grid[None, :], device=device)
    sc_d = simulate(sde, z0, torch.as_tensor([d_ + 1], device=device), dd, 2000, gen=gen)[0].cpu().numpy()
    sc_0 = simulate(sde, z0, torch.zeros(1, dtype=torch.long, device=device), dd, 2000, gen=gen)[0].cpu().numpy()
    C_WITH, C_WITHOUT, C_REAL, C_PAST, C_AX = "#3987E5", "#D95926", "#FFFFFF", "#8A94A6", "#C9D1DC"
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(13, 5.4), facecolor=bg)
    ax.set_facecolor(bg)
    m0, md = np.median(sc_0, 0), np.median(sc_d, 0)
    ax.fill_between(x, np.percentile(sc_d, 25, 0), np.percentile(sc_d, 75, 0), color=C_WITH, alpha=0.25, lw=0)
    ax.fill_between(x, md, m0, where=m0 > md, color=C_WITHOUT, alpha=0.30, lw=0)  # the estimated benefit of the drug
    ax.plot(x, m0, color=C_WITHOUT, lw=3, ls="--")
    ax.plot(x, md, color=C_WITH, lw=3)
    gap = max(0.0, 0.14 * max(sv.max(), m0.max()) - (m0[-1] - md[-1])) / 2  # keep the two labels (two lines each) apart
    ax.text(x[-1] + 0.3, m0[-1] + gap, "without the drug\n(model estimate)", color=C_WITHOUT, fontsize=16, va="center")
    ax.text(x[-1] + 0.3, md[-1] - gap, f"with {DRUGS[d_]}\n(forecast)", color=C_WITH, fontsize=16, va="center")
    past, fut = (sd <= 0) & (sd > -12), (sd > 0) & (sd <= 12)
    ax.scatter(sd[past], sv[past], s=55, color=C_PAST, zorder=3, label="before the first dose")
    ax.scatter(sd[fut], sv[fut], s=75, color=C_REAL, edgecolor=bg, lw=1.2, zorder=4, label="what really happened")
    ax.axvline(0, color=C_PAST, lw=1.2, ls=":")
    top = max(sv[past | fut].max(), np.percentile(sc_0, 75, 0).max())
    ax.text(0.2, top * 1.12, "first dose", color=C_AX, fontsize=14)
    ax.set_ylim(0, top * 1.2)
    ax.set_xlim(-12, 16.5)
    ax.set_xticks([-12, -6, 0, 6, 12])
    ax.set_xlabel("months", color=C_AX)
    ax.set_ylabel("disease score", color=C_AX)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#3A4556")
    ax.tick_params(colors=C_AX)
    ax.grid(axis="y", color="#253041", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=14, labelcolor=C_AX)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=bg)
    plt.close(fig)
    print(f"S: {pid} {DRUGS[d_]}; months 0-12 median with the drug {md.mean():.1f}, without {m0.mean():.1f}")


DARK = {"bg": "#111827", "ax": "#C9D1DC", "grid": "#253041", "spine": "#3A4556", "past": "#8A94A6", "real": "#FFFFFF", "with": "#3987E5",
        "without": "#D95926"}


def dark_style(ax):
    ax.set_facecolor(DARK["bg"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(DARK["spine"])
    ax.tick_params(colors=DARK["ax"])
    ax.grid(axis="y", color=DARK["grid"], lw=0.8)
    ax.set_axisbelow(True)


def fig_columns(sde, data, Z, device, gen, out_dir, pid_cf="HS030725", size=(5.0, 3.9)):
    """Two small plots for a three-column slide (non-technical, dark; rna_pc1_plot.py --columns makes the third): col1_forecast.png, a selected
    candidate of pitch_picks (every visit inside the 90% band, a change >= 3, at least 6 visits spread over >= 240 days, the predicted change of
    the visit mean in the same direction, >= 3 and within 25% of the observed one, then the smallest relative error): the forecast from a past
    visit against the visits of the next 12 months; col2_whatif.png, patient D of
    F_counterfactual_examples.png from the day before the start, with the drug and without any biologic (shaded: the estimated benefit)."""
    D = DARK
    plt.rcParams.update({"font.size": 16})
    grid = np.arange(0, 366, 7)
    x = grid / MONTH
    dd = torch.as_tensor(grid[None, :], device=device)

    def scores(i, k):
        s0, s1 = data.off[i], data.off[i + 1]
        hs = data.has_score[s0:s1]
        sd = (data.day[s0:s1][hs] - data.day[s0 + k]) / MONTH
        return sd, data.score[s0:s1][hs]

    def finish(ax, top, xlabels, name):
        ax.set_ylim(0, top)
        ax.set_xlim(-12.5, 12.5)
        ax.set_xticks([-12, 0, 12])
        ax.set_xticklabels(xlabels, fontsize=14)
        ax.set_ylabel("disease score (higher = worse)", color=D["ax"], fontsize=14)
        ax.axvline(0, color=D["past"], lw=1.2, ls=":")
        dark_style(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=13, labelcolor=D["ax"], ncol=2, handlelength=1.4, columnspacing=1.0)
        ax.figure.tight_layout()
        ax.figure.savefig(f"{out_dir}/{name}", dpi=200, facecolor=D["bg"])
        plt.close(ax.figure)

    # 1. the forecast
    picks, res = pitch_picks(sde, data, Z, device, gen)
    good = [r for r in res if r[1] == 1.0 and r[2] >= 3 and len(r[3][3]) >= 6 and r[3][3].max() - r[3][3].min() >= 240]
    shown = []  # the forecast must show the change: predicted and observed mean of the visits move the same way from the last 6 months, >= 3,
    # and the predicted change is within 25% of the observed one
    for r in good:
        i, kk, plan, fd, fv = r[3]
        sd, sv = scores(i, kk)
        pm = sv[(sd <= 0) & (sd > -6)].mean()
        med = np.median(simulate(sde, Z[[data.off[i] + kk]], torch.as_tensor([plan], device=device), torch.as_tensor(fd[None, :], device=device), 400,
                                 gen=gen)[0].cpu().numpy(), 0)
        dp, do = med.mean() - pm, fv.mean() - pm
        print(f"   {data.pid[i]} plan {plan}: rel. error {r[0]:.2f}, observed change {do:+.1f}, predicted change {dp:+.1f}")
        if np.sign(dp) == np.sign(do) and abs(dp) >= 3 and abs(dp - do) <= 0.25 * abs(do):
            shown.append(r)
    _, _, _, (i, kk, plan, _, _) = min(shown or good, key=lambda r: r[0]) if good else picks[1]
    sd, sv = scores(i, kk)
    sc = simulate(sde, Z[[data.off[i] + kk]], torch.as_tensor([plan], device=device), dd, 2000, gen=gen)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=size, facecolor=D["bg"])
    past, fut = (sd <= 0) & (sd > -12), (sd > 0) & (sd <= 12)
    ax.fill_between(x, np.percentile(sc, 5, 0), np.percentile(sc, 95, 0), color=D["with"], alpha=0.25, lw=0)
    ax.plot(x, np.median(sc, 0), color=D["with"], lw=3, label="forecast")
    ax.scatter(sd[past], sv[past], s=40, color=D["past"], zorder=3, label="before today")
    ax.scatter(sd[fut], sv[fut], s=60, color=D["real"], edgecolor=D["bg"], lw=1, zorder=4, label="what happened")
    finish(ax, 1.45 * max(sv[past | fut].max(), np.percentile(sc, 95, 0).max()), ["a year ago", "today", "in a year"], "col1_forecast.png")
    print(f"K1: {len(good)} candidates meet the rule, {len(shown)} also show the change; {data.pid[i]} plan {plan}")
    # 2. with and without the drug
    i = int(np.flatnonzero(data.pid == pid_cf)[0])
    s0, s1 = data.off[i], data.off[i + 1]
    st, en, dr = data.bio[i]
    kd = next(int(k) for k in np.flatnonzero(data.kind[s0:s1] == DECISION) if (st == data.day[s0 + k] + 1).any())
    d_ = int(dr[np.flatnonzero(st == data.day[s0 + kd] + 1)[0]])
    sd, sv = scores(i, kd)
    z0 = Z[[s0 + kd]]
    sc_d = simulate(sde, z0, torch.as_tensor([d_ + 1], device=device), dd, 2000, gen=gen)[0].cpu().numpy()
    sc_0 = simulate(sde, z0, torch.zeros(1, dtype=torch.long, device=device), dd, 2000, gen=gen)[0].cpu().numpy()
    m0, md = np.median(sc_0, 0), np.median(sc_d, 0)
    fig, ax = plt.subplots(figsize=size, facecolor=D["bg"])
    past, fut = (sd <= 0) & (sd > -12), (sd > 0) & (sd <= 12)
    ax.fill_between(x, md, m0, where=m0 > md, color=D["without"], alpha=0.35, lw=0)
    ax.plot(x, m0, color=D["without"], lw=3, ls="--", label="without the drug")
    ax.plot(x, md, color=D["with"], lw=3, label="with the drug")
    ax.scatter(sd[past], sv[past], s=40, color=D["past"], zorder=3, label="before")
    ax.scatter(sd[fut], sv[fut], s=60, color=D["real"], edgecolor=D["bg"], lw=1, zorder=4, label="what happened")
    finish(ax, 1.45 * max(sv[past | fut].max(), m0.max()), ["a year before", "first dose", "a year after"], "col2_whatif.png")
    print(f"K2: {pid_cf} {DRUGS[d_]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sde", default=None, help="weights of the SDE (default: <run>/sde.pt)")
    ap.add_argument("--out", default=None, help="folder of the plots (default: <run>/sde_plots)")
    ap.add_argument("--figs", default="ABC", help="plots to make: A, B, C, P (selected examples) and F (selected counterfactual examples)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, _ = load(args.run, device)
    Z = torch.as_tensor(np.load(f"{args.run}/sde_cache.npz")["Z"], device=device)
    sde = LatentSDE().to(device)
    sde.load_state_dict(torch.load(args.sde or f"{args.run}/sde.pt", map_location=device))
    sde.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    out = args.out or f"{args.run}/sde_plots"
    os.makedirs(out, exist_ok=True)
    if "A" in args.figs:
        fig_patients(sde, model, data, Z, device, gen, f"{out}/A_patients.png")
    if "B" in args.figs:
        n, m_sde, m_h2, cov = fig_calibration(sde, model, data, Z, device, gen, f"{out}/B_calibration.png")
        print(f"B: {n} pairs; MAE SDE median {m_sde:.3f}, head 2 {m_h2:.3f}; coverage {cov}")
    if "C" in args.figs:
        fig_drug_paths(sde, data, Z, device, gen, f"{out}/C_drug_paths.png")
    if "P" in args.figs:
        fig_pitch(sde, model, data, Z, device, gen, f"{out}/P_selected_examples.png")
    if "F" in args.figs:
        fig_counterfactual(sde, model, data, Z, device, gen, f"{out}/F_counterfactual_examples.png")
    if "S" in args.figs:
        fig_simple(sde, data, Z, device, gen, f"{out}/S_one_patient_dark.png")
    if "K" in args.figs:
        fig_columns(sde, data, Z, device, gen, out)
    print(f"plots in {out}")


if __name__ == "__main__":
    main()
