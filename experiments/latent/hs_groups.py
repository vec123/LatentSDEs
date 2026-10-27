"""
Responder groups from the predicted response, checked on held-out starts.

For each biologic start of a held-out patient (decision step = the day before): the predicted response Y_d / Y0 from head 2 at the
decision step (window 90..270 days after the start, drug d against plan "none"; the scale of the visit-based labels). The realized
response is the mean score in days 90..270 after the start divided by the mean in the 180 days before (as in rna_response.py; only starts
with both windows and a pre mean >= 3). Good responder: realized ratio < 0.5.
Groups: predicted ratio < 0.5 / 0.5-0.8 / >= 0.8 (and terciles). Per group: the share of good responders and the mean realized ratio, with a
95% interval from a bootstrap over patients. A group counts only if the realized response differs between the groups on the held-out starts.
Profiles: the standardized mean difference of the start features and (RNAseq starts) the RNAseq components, group "predicted good" against
the others. The data are observational: a group is a hypothesis, not a causal drug effect.

    python -m experiments.latent.hs_groups runs/hs_e2e_<stamp>
"""
import argparse

import numpy as np
import torch

from experiments.latent.hs_task2 import DECISION, DRUGS, MONTH, PROFILE, auc, load, plan_at


def find_starts(data, pats):
    out = []
    for i in pats:
        s0, s1 = data.off[i], data.off[i + 1]
        st, en, dr = data.bio[i]
        hs = data.has_score[s0:s1]
        sd, sv = data.day[s0:s1][hs], data.score[s0:s1][hs]
        for kd in np.flatnonzero(data.kind[s0:s1] == DECISION):
            s = data.day[s0 + kd] + 1
            j = np.flatnonzero(st == s)
            if not len(j):
                continue
            pre, post = sv[(sd >= s - 180) & (sd < s)], sv[(sd >= s + 90) & (sd <= s + 270)]
            if len(pre) and len(post) and pre.mean() >= 3:
                out.append((int(i), int(s), int(dr[j[0]]), int(kd), post.mean() / pre.mean()))
    return out


@torch.no_grad()
def predicted_ratio(model, data, starts, device, **hide):
    """Y_d / Y0 for the given drug and Y_d / Y0 for all 3 drugs, head 2 at the decision step"""
    ys, ym, given, allr = data.stats["ystd"], data.stats["ymean"], [], []
    for a in range(0, len(starts), 50):
        part = starts[a:a + 50]
        h = model.encode(data.pack([p[0] for p in part], device, **hide))
        n = len(part)
        bi, kd = torch.arange(n, device=device), torch.as_tensor([p[3] for p in part], device=device)
        hz = torch.full((n,), (90 + 1) / MONTH, device=device)
        tau = torch.full((n,), 90 / MONTH, device=device)
        y0 = (plan_at(model, h, bi, kd, hz, torch.zeros(n, device=device), torch.zeros(n, dtype=torch.long, device=device))[0] * ys + ym).clamp(min=0.5)
        yd = torch.stack([plan_at(model, h, bi, kd, hz, tau, torch.full((n,), d + 1, dtype=torch.long, device=device))[0] * ys + ym
                          for d in range(len(DRUGS))], 1)
        r = (yd / y0[:, None]).cpu().numpy()
        allr.append(r)
        given.append(r[np.arange(n), [p[2] for p in part]])
    return np.concatenate(given), np.concatenate(allr)


def boot(fn, pat, rng, n=1000):
    """95% interval of fn(index array) with a bootstrap over patients"""
    u, inv = np.unique(pat, return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(u))]
    v = [fn(np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(n)]
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def report(title, pred, real, pat, rng):
    good = real < 0.5
    print(f"\n{title}: {len(pred)} starts, {len(np.unique(pat))} patients; good responders {good.mean():.3f}; AUC of -predicted ratio {auc(-pred, good):.3f}")
    groups = {"predicted < 0.5": pred < 0.5, "predicted 0.5-0.8": (pred >= 0.5) & (pred < 0.8), "predicted >= 0.8": pred >= 0.8}
    q = np.quantile(pred, [1 / 3, 2 / 3])
    groups.update({"tercile 1 (lowest pred.)": pred <= q[0], "tercile 2": (pred > q[0]) & (pred <= q[1]), "tercile 3 (highest pred.)": pred > q[1]})
    print(f"   {'group':26s} {'n':>5s} {'pred. ratio':>11s} {'real good share':>16s} {'95% interval':>16s} {'real mean ratio':>16s}")
    for g, s in groups.items():
        if s.sum() < 10:
            continue
        lo, hi = boot(lambda ix: good[ix][s[ix]].mean() if s[ix].any() else np.nan, pat, rng)
        print(f"   {g:26s} {s.sum():5d} {pred[s].mean():11.2f} {good[s].mean():16.3f} {f'[{lo:.3f}, {hi:.3f}]':>16s} {np.median(real[s]):16.2f}")
    lo_g, hi_g = groups["tercile 1 (lowest pred.)"], groups["tercile 3 (highest pred.)"]
    diff = lambda ix: good[ix][lo_g[ix]].mean() - good[ix][hi_g[ix]].mean()
    lo, hi = boot(diff, pat, rng)
    print(f"   good share, tercile 1 - tercile 3: {diff(np.arange(len(pred))):+.3f} [{lo:+.3f}, {hi:+.3f}]  (the groups count if this excludes 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, _ = load(args.run, device)
    rng = np.random.default_rng(0)
    starts = find_starts(data, np.flatnonzero(data.flag))
    pat = np.array([s[0] for s in starts])
    drug = np.array([s[2] for s in starts])
    real = np.array([s[4] for s in starts])
    kd_row = np.array([data.off[s[0]] + s[3] for s in starts])
    rna = data.rna_row[kd_row] >= 0
    pred, allr = predicted_ratio(model, data, starts, device)
    report("all held-out starts (full input)", pred, real, pat, rng)
    for d in range(len(DRUGS)):
        if (drug == d).sum() >= 60:
            report(f"drug {DRUGS[d]}", pred[drug == d], real[drug == d], pat[drug == d], rng)
    if rna.sum() >= 20:
        report("starts with a baseline RNAseq sample", pred[rna], real[rna], pat[rna], rng)
        p_h, _ = predicted_ratio(model, data, starts, device, hide_rna=True)
        report("same RNAseq starts, RNAseq hidden", p_h[rna], real[rna], pat[rna], rng)

    best = np.argmin(allr, 1)
    same = best == drug
    print(f"\ndrug choice (observational, confounded): the given drug is the predicted best for {same.mean():.2f} of the starts")
    print(f"   realized good share: given = predicted best {(real[same] < 0.5).mean():.3f} (n={same.sum()}), other {(real[~same] < 0.5).mean():.3f} (n={(~same).sum()})")

    adjusted(data, kd_row, drug, pred, real, pat, rng, H=states(model, data, starts, device))

    good_pred = pred < np.quantile(pred, 1 / 3)
    print("\nprofile at the start: lowest predicted-ratio tercile against the rest (standardized mean difference)")
    smd = []
    for c in PROFILE:
        j = data.names.index(c)
        v = np.where(data.M[kd_row, j], data.raw[kd_row, j], np.nan)
        sd = np.nanstd(v)
        if sd > 0:
            smd.append((c, (np.nanmean(v[good_pred]) - np.nanmean(v[~good_pred])) / sd, np.nanmean(v[good_pred]), np.nanmean(v[~good_pred])))
    for k in range(len(DRUGS)):
        smd.append((f"drug = {DRUGS[k]}", (np.mean(drug[good_pred] == k) - np.mean(drug[~good_pred] == k)) / max(np.std(drug == k), 1e-6),
                    np.mean(drug[good_pred] == k), np.mean(drug[~good_pred] == k)))
    for c, s_, a, b in sorted(smd, key=lambda r: -abs(r[1]))[:12]:
        print(f"   {c:26s} {s_:+.2f}   ({a:.2f} vs {b:.2f})")
    if rna.sum() >= 20 and (good_pred & rna).sum() >= 5:
        E = data.rna_emb[data.rna_row[kd_row[rna]]]
        g = good_pred[rna]
        smd = [(k, (E[g, k].mean() - E[~g, k].mean()) / max(E[:, k].std(), 1e-6)) for k in range(E.shape[1])]
        print(f"   RNAseq components (principal components, {rna.sum()} starts, {g.sum()} in the group):")
        print("   " + "  ".join(f"PC{k + 1} {s_:+.2f}" for k, s_ in sorted(smd, key=lambda r: -abs(r[1]))[:6]))


@torch.no_grad()
def states(model, data, starts, device):
    """the GRU state at the decision step of each start (full input)"""
    out = []
    for a in range(0, len(starts), 50):
        part = starts[a:a + 50]
        h = model.encode(data.pack([p[0] for p in part], device))
        out.append(h[torch.arange(len(part), device=device), torch.as_tensor([p[3] for p in part], device=device)].cpu().numpy())
    return np.concatenate(out)


def adjusted(data, kd_row, drug, pred, real, pat, rng, H=None):
    """does the predicted ratio add to the baseline severity? 1. split inside strata of the recent mean score; 2. out-of-fold AUC of a
    logistic model on the start features alone, and with the predicted ratio added (5 folds, grouped by patient)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    good = real < 0.5
    col = lambda c: np.where(data.M[kd_row, data.names.index(c)], data.raw[kd_row, data.names.index(c)], np.nan)
    sev = col("mean_6m")
    sev = np.where(np.isnan(sev), np.nanmedian(sev), sev)
    print("\nadjusted check 1: inside strata of the recent mean score (mean_6m), predicted ratio above / below the stratum median")
    cut = np.quantile(sev, [0.25, 0.5, 0.75])
    stratum = np.searchsorted(cut, sev)
    low = np.zeros(len(pred), bool)
    for k in range(4):
        s = stratum == k
        low[s] = pred[s] <= np.median(pred[s])
        print(f"   stratum {k + 1} (mean_6m {sev[s].min():.1f}..{sev[s].max():.1f}, n={s.sum()}): good share low predicted ratio {good[s & low].mean():.3f}, "
              f"high {good[s & ~low].mean():.3f}")
    def pooled(ix):
        d = [good[ix][(stratum[ix] == k) & low[ix]].mean() - good[ix][(stratum[ix] == k) & ~low[ix]].mean() for k in range(4)]
        return float(np.nanmean(d))
    lo, hi = boot(pooled, pat, rng)
    print(f"   pooled difference (low - high predicted ratio): {pooled(np.arange(len(pred))):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    names = ["hurley_stage_baseline", "mean_all", "mean_6m", "mean_12m", "last_score", "n_scores_all", "n_surg", "days_since_diag", "max_bio_line",
             "age_at_visit", "bmi_at_baseline", "sex", "smoking_status", "diagnostic_delay_years", "Antibiotic_active", "vis6m_flare_reactive"]
    B = np.column_stack([col(c) for c in names] + [drug == k for k in range(len(DRUGS))]).astype(float)
    B = np.where(np.isnan(B), np.nanmedian(B, 0), B)
    models = {"start features alone": B, "predicted ratio alone": pred[:, None], "start features + predicted ratio": np.column_stack([B, pred])}
    if H is not None:
        models.update({"GRU state alone": H, "start features + GRU state": np.column_stack([B, H]),
                       "start features + GRU state + ratio": np.column_stack([B, H, pred])})
    oof = {k: np.zeros(len(pred)) for k in models}
    for tr_, te_ in GroupKFold(5).split(B, good, pat):
        for k, X in models.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(X[tr_], good[tr_])
            oof[k][te_] = m.predict_proba(X[te_])[:, 1]
    print("\nadjusted check 2: out-of-fold AUC for good responders (logistic, 5 folds grouped by patient)")
    for k, v in oof.items():
        print(f"   {k:34s} {auc(v, good):.3f}")
    f = lambda ix: auc(oof["start features + predicted ratio"][ix], good[ix]) - auc(oof["start features alone"][ix], good[ix])
    lo, hi = boot(f, pat, rng, n=300)
    print(f"   gain of the predicted ratio over the start features: {f(np.arange(len(pred))):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if H is not None:
        f = lambda ix: auc(oof["start features + GRU state"][ix], good[ix]) - auc(oof["start features alone"][ix], good[ix])
        lo, hi = boot(f, pat, rng, n=300)
        print(f"   gain of the GRU state over the start features: {f(np.arange(len(pred))):+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
