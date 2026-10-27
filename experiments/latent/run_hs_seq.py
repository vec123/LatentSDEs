"""
Auto-decoder latent SDE on hs_sequences_train.npz, predicting only the target y(t) = mean IHS4 score in (t, t + 6 months].

Input at each visit: all 89 feature values (standardized, 0 where missing) and their 89 mask flags, concatenated, as the
GRU input of the forward posterior (the standardized time since the first visit is appended). The posterior at visit t reads
the visits up to t, never y and never a later visit. The decoder p(y | z) reads z alone; y is only the target of the
reconstruction term. A learned code per patient (embedding table, N(0, I) prior) goes in as mu (posterior, drift, diffusion),
as in run_severity.py.

Training: the patients with is_test_patient = 0. Test: the 4,120 flagged patients, never trained on. A step t is scored with
the history up to t only: the code is fitted (model frozen) on the rows up to t, and a y label of the history counts only if
its whole window is before t (row time <= t - 6 months). The prediction is the decoded posterior mean at t. Two kinds of
step: the last step of each patient (y_is_real = 1, the official rows) and one random derived step per patient.
Baselines: train mean of y, last score, mean of the past 6 months (the npz features last_score, mean_6m), and the SDE with
code 0 (no fit: what the code adds).

    python -m experiments.latent.run_hs_seq [--seconds 600] [--steps 15000] [--batch 25]

Run it with environments/torch_gpu_env (CUDA); keep the batch at 25 and no CPU-heavy work beside it (the laptop throttles).
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml

from experiments.latent.compare_baselines import metrics
from experiments.latent.run import build_model

LIGHT = "test_data/hs_public_dataset/hs_sequences_all_light.npz"  # arrays of hs_sequences_train.npz, ids as str; --data: the one-hot version (make_onehot_npz.py)
T_SCALE = 30.0
CFG = {
    "dim_z": 8, "dim_mu": 8, "dim_u": 0, "dim_feat": 89,
    "encoder": {"type": "none"},
    "decoder": {"type": "gaussian", "hidden": [32], "std_init": 0.5, "reads_mu": False},
    "posterior": {"type": "sequence", "hidden": 64, "bidirectional": False, "forward_only": True},
    "drift": {"hidden": [64, 64]},
    "diffusion": {"type": "constant"},
    "g_init": 0.3, "n_samples": 4, "n_quad": 32,
}


# Engineered summaries of the score and visit history (computed by the data provider from the visits). --drop_engineered removes them, so that the
# model sees one visit at a time (its own score, its time and gap, treatments, surgeries, statics, images, RNA) and must read the history itself.
ENGINEERED = ["mean_3m", "n_3m", "std_3m", "mean_6m", "n_6m", "std_6m", "mean_12m", "n_12m", "std_12m", "mean_prev6m", "n_prev6m", "trend_6m",
              "mean_all", "n_scores_all", "last_score", "days_since_score",
              "vis6m_background", "vis6m_flare_reactive", "vis6m_follow_up", "vis6m_treatment_monitoring"]
EXTRA_NAMES = ["ewma_1m", "ewma_3m", "ewma_12m", "max_6m", "min_6m", "prev_score", "prev2_score"]


def extra_features(t, x, o):
    """
    Causal time-series features of one patient from its score history. t (n,) visit times in months, x (n,) the score of the visit
    (own_score), o (n,) True where a score was taken. Row k reads the scores at rows <= k only. Returns (n, 7) values and masks:
    time-decayed means of the scores (time constants 1, 3 and 12 months; the weight of a score is exp(-age / tau)), the max and the
    min of the scores of the last 6 months, and the two scores before the latest one.
    """
    n, taus = len(t), np.array([1.0, 3.0, 12.0])
    out, mask = np.zeros((n, 7), np.float32), np.zeros((n, 7), bool)
    num, den, last_t, ht, hx = np.zeros(3), np.zeros(3), None, [], []
    for k in range(n):
        if o[k]:
            if last_t is not None:
                d = np.exp(-(t[k] - last_t) / taus)
                num, den = num * d, den * d
            num, den, last_t = num + x[k], den + 1.0, t[k]
            ht.append(t[k])
            hx.append(x[k])
        if ht:
            out[k, :3], mask[k, :3] = num / den, True  # a ratio: the decay between scores cancels
            w = np.asarray(ht) > t[k] - 6.0
            if w.any():
                v = np.asarray(hx)[w]
                out[k, 3], out[k, 4], mask[k, 3:5] = v.max(), v.min(), True
            if len(hx) >= 2:
                out[k, 5], mask[k, 5] = hx[-2], True
            if len(hx) >= 3:
                out[k, 6], mask[k, 6] = hx[-3], True
    return out, mask


class Data:
    """The npz as per-patient slices of standardized features, masks, targets and times."""

    def __init__(self, path=LIGHT, extra=False, treatment_u=False, drop_engineered=False, real_flag=False):
        d = np.load(path)
        self.names, static = list(d["feature_names"]), d["static_feature"].astype(bool)
        self.off, self.pid, self.flag = d["offsets"], d["patient_id"], d["is_test_patient"].astype(bool)
        X, self.M = d["X"].copy(), d["M"] > 0
        self.raw = {n: (X[:, self.names.index(n)].copy(), self.M[:, self.names.index(n)]) for n in ("last_score", "mean_6m", "own_score")}
        self.months = d["time_months"]
        self.U = None
        if treatment_u:  # the treatment state of each visit, the forcing u(t) of the drift (past status only; raw values, before the transforms)
            col = lambda n: X[:, self.names.index(n)]
            g = lambda n: np.where(self.M[:, self.names.index(n)], col(n), 0.0)
            has_surg = self.M[:, self.names.index("days_since_surg")]
            on = g("n_tx_active") > 0
            self.U = np.stack([
                g("Biologic_active"),                                          # a biologic is active
                g("Biologic_active") * (g("max_bio_line") >= 2),               # ... and a second or later line was reached
                np.clip(g("Antibiotic_active"), 0, 2) / 2,                     # antibiotics active
                np.clip(g("n_tx_active"), 0, 4) / 4,                           # number of active treatments
                np.where(has_surg, np.exp(-g("days_since_surg") / 60.0), 0.0),  # a recent surgery (fades with a 60-day constant)
                np.where(on, np.log1p(g("days_since_tx_start")) / 8.0, 0.0),   # time on the latest started treatment
                np.clip(g("Biologic_ever"), 0, 2) / 2,                         # prior biologic exposure
            ], 1).astype(np.float32)
        if extra:  # causal time-series features of the score history appended to the 89 (extra_features)
            own, own_m = X[:, self.names.index("own_score")], self.M[:, self.names.index("own_score")]
            E, EM = np.zeros((len(X), len(EXTRA_NAMES)), np.float32), np.zeros((len(X), len(EXTRA_NAMES)), bool)
            for i in range(len(self.pid)):
                a, b = self.off[i], self.off[i + 1]
                E[a:b], EM[a:b] = extra_features(self.months[a:b], own[a:b], own_m[a:b])
            X, self.M = np.concatenate([X, E], 1), np.concatenate([self.M, EM], 1)
            self.names = self.names + EXTRA_NAMES
            static = np.concatenate([static, np.zeros(len(EXTRA_NAMES), bool)])
        self.y, self.ym, self.yr = d["y"], d["y_mask"] > 0, d["y_is_real"] > 0
        train = np.repeat(~self.flag, np.diff(self.off))
        for j, n in enumerate(self.names):  # log1p of the long-tailed non-negative columns (counts, days since)
            if static[j] or n == "time_months":
                continue
            o = self.M[:, j]
            if o.any() and X[o, j].min() >= 0 and X[o, j].max() > 50:
                X[:, j] = np.where(o, np.log1p(np.maximum(X[:, j], 0)), 0)
        if drop_engineered:  # the score-history summaries leave the inputs (the baselines and targets keep reading self.raw)
            keep = [j for j, n in enumerate(self.names) if n not in ENGINEERED]
            X, self.M, static = X[:, keep], self.M[:, keep], static[keep]
            self.names = [self.names[j] for j in keep]
        fit = train[:, None] & self.M
        n_obs = fit.sum(0).clip(min=1)
        mean = (np.where(fit, X, 0).sum(0) / n_obs)
        std = np.sqrt((np.where(fit, (X - mean) ** 2, 0)).sum(0) / n_obs)
        std[std < 1e-6] = 1.0
        self.X = np.where(self.M, (X - mean) / std, 0).astype(np.float32)
        yt = self.y[train & self.ym]
        self.ymean, self.ystd = float(yt.mean()), float(yt.std())
        self.train_mean_y = self.ymean
        self.train_idx = np.flatnonzero(~self.flag)
        self.test_idx = np.flatnonzero(self.flag)
        if real_flag:  # one more input: 1 at the row whose label is the real one (the last visit), 0 at the other rows (as the is_real column of the XGBoost table)
            self.X = np.concatenate([self.X, self.yr[:, None].astype(np.float32)], 1)
            self.M = np.concatenate([self.M, np.ones((len(self.X), 1), bool)], 1)
            self.names = self.names + ["is_real_flag"]
        # weight of the label of each row: 1 for a real label; 1 / sqrt(number of derived labels of the patient) for a derived one
        self.wrow = np.ones(len(self.X), np.float32)
        for i in range(len(self.pid)):
            a, b = self.off[i], self.off[i + 1]
            der = self.ym[a:b] & ~self.yr[a:b]
            self.wrow[a:b] = np.where(der, 1.0 / np.sqrt(max(der.sum(), 1)), 1.0)
        self.use_w = False

    def n_rows(self, i):
        return int(self.off[i + 1] - self.off[i])

    def horizon_targets(self, H=12, window=6.0):
        """
        (rows, H + 1): the mean score in (t + h, t + h + 6 months] for every visit t and h = 0..H, from the scores of the visits
        (own_score): the definition of y, which this reproduces at h = 0 (checked: max difference 1.5e-6). NaN where the window
        holds no score or ends after the patient's last visit. A target is a future value of the training patient; it is never an input.
        """
        own, own_m = self.raw["own_score"]
        T = np.full((len(self.months), H + 1), np.nan, np.float32)
        steps = np.arange(H + 1)[None, :]
        for i in range(len(self.pid)):
            a, b = self.off[i], self.off[i + 1]
            t, o = self.months[a:b], own_m[a:b]
            ts, xs = t[o], own[a:b][o]
            if len(ts) == 0:
                continue
            cs = np.concatenate([[0.0], np.cumsum(xs)])
            lo = t[:, None] + steps
            hi = lo + window
            ia, ib = np.searchsorted(ts, lo, side="right"), np.searchsorted(ts, hi, side="right")
            cnt = ib - ia
            v = (cs[ib] - cs[ia]) / np.maximum(cnt, 1)
            T[a:b] = np.where((cnt > 0) & (hi <= t[-1]), v, np.nan)
        return T

    def pack(self, items, code, device, only_last=False, real_only=False):
        """
        items: list of (patient index, number of rows used, cutoff in months or None): the first n rows of the patient; the
        y label of a row is visible only if its time <= cutoff (None: all). code (B, P). One batch dict: t, x = y target,
        mask = y mask, feat, feat_mask for the posterior, mu = code. only_last: only the label of the last used row counts.
        real_only: only the real labels (y_is_real, the last visit of each patient) count.
        """
        B, N = len(items), max(2, max(n for _, n, _ in items))
        D = self.X.shape[1]
        t, x, m = np.zeros((B, N)), np.zeros((B, N, 1)), np.zeros((B, N, 1))
        feat, fm = np.zeros((B, N, D), np.float32), np.zeros((B, N, D), np.float32)
        F = 0 if self.U is None else self.U.shape[1]
        u = np.zeros((B, N, F), np.float32)
        for b, (i, n, cut) in enumerate(items):
            s = slice(self.off[i], self.off[i] + n)
            if F:
                u[b, :n], u[b, n:] = self.U[s], self.U[s][-1]
            tb = self.months[s] / T_SCALE + 1e-4 * np.arange(n)  # strictly increasing
            t[b, :n], t[b, n:] = tb, tb[-1]
            seen = self.ym[s] if cut is None else self.ym[s] & (self.months[s] <= cut)
            if only_last:
                seen = seen & (np.arange(n) == n - 1)
            if real_only:
                seen = seen & self.yr[s]
            x[b, :n, 0], m[b, :n, 0] = np.where(seen, (self.y[s] - self.ymean) / self.ystd, 0), seen
            if self.use_w:
                m[b, :n, 0] = seen * self.wrow[s]
            feat[b, :n], fm[b, :n] = self.X[s], self.M[s]
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        return {"t": f(t), "x": f(x), "mask": f(m), "feat": f(feat), "feat_mask": f(fm), "mu": code, "u": f(u)}


def code_prior(c):
    return 0.5 * c.pow(2).sum()


def fit_codes(model, data, items, device, steps=60, lr=0.05, beta=1.0):
    """
    Codes for patients that are given only a history: Adam on the codes from 0 (N(0, I) prior), the model weights untouched
    (the gradient is taken with respect to the codes alone). Returns (codes, ok): ok is False for a patient whose loss is not
    finite (e.g. a history of near-identical times). Each patient's loss is its own, so a bad patient cannot spoil the
    others: its terms are left out and its code stays where it was. The result carries no graph.
    """
    c = torch.zeros(len(items), CFG["dim_mu"], device=device, requires_grad=True)
    batch = data.pack(items, c, device)
    m, v = torch.zeros_like(c), torch.zeros_like(c)
    model.train()  # cuDNN GRU backward needs train mode; the model has no dropout
    ok = torch.ones(len(items), dtype=torch.bool, device=device)
    for k in range(1, steps + 1):
        batch["mu"] = c
        per, _ = model.loss(batch, beta=beta, per_trajectory=True)  # (B,): the negative ELBO of each patient
        ok = torch.isfinite(per)
        n = batch["mask"].sum().clamp(min=1)
        g, = torch.autograd.grad(torch.where(ok, per, torch.zeros_like(per)).sum() / n + code_prior(c) / n, c)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        c = (c - lr * (m / (1 - 0.9 ** k)) / ((v / (1 - 0.999 ** k)).sqrt() + 1e-8)).detach().requires_grad_(True)
    ok = ok & torch.isfinite(c).all(1)
    return c.detach(), ok


@torch.no_grad()
def predict(model, data, items, code, device):
    """Decoded posterior mean at the last used row of each patient, in original units."""
    model.eval()
    batch = data.pack(items, code, device)
    model.posterior.condition({**batch, "x": batch["feat"], "mask": batch["feat_mask"]})
    m = model.posterior(batch["t"][:, -1:])[0][:, 0]
    return (model.decoder(m, code).mean[:, 0] * data.ystd + data.ymean).cpu().numpy()


def evaluate(model, data, device, seed=0, max_patients=10 ** 9, chunk=50, beta=1.0, pause=0.0, code_default=None, fit_steps=60, fit_lr=0.05):
    rng = np.random.default_rng(seed)
    kinds = {"real": [], "derived": []}
    for i in data.test_idx[:max_patients]:
        s = slice(data.off[i], data.off[i + 1])
        last = np.flatnonzero(data.yr[s] & data.ym[s])
        der = np.flatnonzero(data.ym[s] & ~data.yr[s] & (data.months[s] > 6))
        if len(last):
            kinds["real"].append((i, int(last[0])))
        if len(der):
            kinds["derived"].append((i, int(rng.choice(der))))
    out = {}
    for kind, rows in kinds.items():
        rows.sort(key=lambda r: r[1])  # similar lengths together
        items = [(i, k + 1, data.months[data.off[i] + k] - 6.0) for i, k in rows]  # labels whose window ends by t
        true = np.array([data.y[data.off[i] + k] for i, k in rows])
        idx = np.array([data.off[i] + k for i, k in rows])
        fits, zeros, n_unfit, n_nan = [], [], 0, 0
        for a in range(0, len(items), chunk):
            part = items[a:a + chunk]
            # a history of one row has no time span (the loss is infinite): no fit, the mean training code instead
            fit = [j for j, (_, n, _) in enumerate(part) if n >= 3]
            c = code_default.to(device).expand(len(part), -1).clone()
            if fit:
                cf, ok = fit_codes(model, data, [part[j] for j in fit], device, steps=fit_steps, lr=fit_lr, beta=beta)
                good = torch.as_tensor(fit, device=device)[ok]
                c[good] = cf[ok]
                n_unfit += len(part) - len(good)
            fits.append(predict(model, data, part, c, device))
            zeros.append(predict(model, data, part, torch.zeros_like(c), device))
            for a_ in (fits, zeros):  # a prediction that is not finite falls back to the train mean of y
                bad = ~np.isfinite(a_[-1])
                n_nan += int(bad.sum())
                a_[-1] = np.where(bad, data.train_mean_y, a_[-1])
            time.sleep(pause)  # lets the laptop GPU cool between chunks
        base = lambda n: np.where(data.raw[n][1][idx], data.raw[n][0][idx], data.train_mean_y)
        preds = {"Train mean": np.full_like(true, data.train_mean_y), "Persistence: last score": base("last_score"),
                 "Persistence: past 6-month mean": base("mean_6m"), "Latent SDE, code 0": np.concatenate(zeros),
                 "Latent SDE, fitted code": np.concatenate(fits)}
        out[kind] = {n: metrics(p, true) for n, p in preds.items()}
        print(f"\n{kind} steps: {len(rows)} patients, y mean {true.mean():.2f} std {true.std():.2f}; {n_unfit} with the "
              f"mean code (history < 3 rows or loss not finite), {n_nan} predictions not finite", flush=True)
        for n, m in out[kind].items():
            print(f"{n:32s} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}", flush=True)
    return out


def forecast_loss(model, data, targets, batch, items, args, device, rng):
    """
    Trains the drift to forecast. Per patient, `fc_cuts` random visits are cuts. The posterior state at a cut (it has read the visits up to the
    cut only) is sampled `fc_samples` times and the prior SDE is simulated 12 months ahead (Euler-Maruyama, `fc_sub` sub-steps per month,
    gradients through the path); at each month h = 1..12 the decoder scores the true mean score of the 6 months after cut + h (targets, standardized).
    Mean negative log-likelihood per scored target. Needs model.loss(batch) to have run on the same batch (the posterior keeps its GRU states).
    """
    B, N = batch["t"].shape
    K, S, H = args.fc_cuts, args.fc_samples, 12
    rows = np.zeros((B, K), np.int64)
    Y = np.full((B, K, H), np.nan, np.float32)
    for b, (i, n, _) in enumerate(items):
        T = targets[data.off[i]:data.off[i] + n, 1:H + 1]
        cand = np.flatnonzero(np.isfinite(T).any(1))
        if len(cand):
            rows[b] = rng.choice(cand, K)
            Y[b] = T[rows[b]]
    Y = (Y - data.ymean) / data.ystd
    valid = torch.as_tensor(np.isfinite(Y), device=device)
    Y = torch.as_tensor(np.nan_to_num(Y), device=device)
    idx = torch.as_tensor(rows, device=device)
    m, logv, _, _ = model.posterior(batch["t"].gather(1, idx))  # (B, K, dim_z): the state at each cut
    z = (m + (0.5 * logv).exp() * torch.randn(S, *m.shape, device=device)).reshape(S * B * K, -1)
    mu = torch.zeros(S * B * K, 0, device=device)
    dt = (1.0 / T_SCALE) / args.fc_sub  # one month is 1 / T_SCALE in the model's time unit
    Uf = None
    if data.U is not None:  # the treatment path the training patient had: u at the start of each sub-step, held between visits
        Uf = np.zeros((B, K, H * args.fc_sub, data.U.shape[1]), np.float32)
        for b, (i, n, _) in enumerate(items):
            mo = data.months[data.off[i]:data.off[i + 1]]
            q = mo[rows[b]][:, None] + np.arange(H * args.fc_sub)[None, :] / args.fc_sub
            Uf[b] = data.U[data.off[i]:data.off[i + 1]][np.clip(np.searchsorted(mo, q, side="right") - 1, 0, len(mo) - 1)]
        Uf = torch.as_tensor(Uf, device=device)
    total = z.new_zeros(())
    for h in range(H):
        for sub in range(args.fc_sub):
            u = mu if Uf is None else Uf[:, :, h * args.fc_sub + sub].unsqueeze(0).expand(S, -1, -1, -1).reshape(S * B * K, -1)
            z = z + model.drift(z, mu, u) * dt + model.diffusion(z, mu, None) * dt ** 0.5 * torch.randn_like(z)
        lp = model.decoder(z, mu).log_prob(Y[..., h].repeat(S, 1, 1).reshape(S * B * K, 1))  # (S B K, 1)
        total = total - (lp.reshape(S, B, K) * valid[..., h]).mean(0).sum()
    return total / valid.sum().clamp(min=1)


def val_mae(model, data, device):
    """MAE on the real-label rows (last visits) of the validation patients, which the training never sees."""
    from experiments.latent.eval_all_rows import predict_rows
    model.eval()
    with torch.no_grad():
        preds = predict_rows(model, data, data.val_idx, device, dim_mu=0)
    err = []
    for i, p in zip(data.val_idx, preds):
        s = slice(data.off[i], data.off[i + 1])
        ok = data.ym[s] & data.yr[s]
        err += list(np.abs(p[ok] - data.y[s][ok]))
    model.train()
    return float(np.mean(err))


def train_all_rows(model, data, args, device, rng, run_dir, log):
    """
    No code. Every step takes 25 whole patients; one forward pass of the posterior gives the state at every visit, and the loss
    counts every labelled visit of the patient (the label y of a visit is its target only, never an input). The posterior is
    forward only, so the state at a visit reads the visits up to it: the training is causal by construction.
    """
    train = data.train_idx
    lengths = np.array([data.n_rows(i) for i in train])
    targets = data.horizon_targets() if args.fc_weight else None
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    start, step = time.time(), 0
    best, best_step, best_state, stop = float("inf"), 0, None, False
    while step < args.steps and time.time() - start < args.seconds and not stop:
        order = np.argsort(lengths + rng.uniform(0, 3, len(lengths)))  # similar lengths together, new groups each epoch
        parts = [order[a:a + args.batch] for a in range(0, len(order), args.batch)]
        for p in rng.permutation(len(parts)):
            if step >= args.steps or time.time() - start >= args.seconds:
                break
            items = [(int(train[k]), data.n_rows(train[k]), None) for k in parts[p]]
            batch = data.pack(items, torch.zeros(len(items), 0, device=device), device, real_only=args.real_only)
            warm = min(1.0, step / 500)
            model.train()
            per, _ = model.loss(batch, beta=args.beta_max * warm, per_trajectory=True)
            good = torch.isfinite(per)
            loss = torch.where(good, per, torch.zeros_like(per)).sum() / batch["mask"].sum().clamp(min=1)
            fc = None
            if args.fc_weight:
                fc = forecast_loss(model, data, targets, batch, items, args, device, rng)
                loss = loss + args.fc_weight * warm * fc
            opt.zero_grad()
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)):
                opt.step()
            sched.step()
            if step % 100 == 0:
                msg = f"{step} t={time.time() - start:.0f}s loss {loss.item():.4f} finite {good.float().mean().item():.2f}" + (
                    f" forecast {fc.item():.4f}" if fc is not None else "")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            step += 1
            if step % 2500 == 0:
                torch.save({"model": model.state_dict()}, f"{run_dir}/model_{step}.pt")
            if len(data.val_idx) and step % args.val_every == 0:
                v = val_mae(model, data, device)
                if v < best:
                    best, best_step, best_state = v, step, {k: x.detach().clone() for k, x in model.state_dict().items()}
                msg = f"{step} validation MAE {v:.4f} (best {best:.4f} at {best_step})"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                if step - best_step >= args.patience * args.val_every:
                    print(f"no improvement for {args.patience} checks: stop; best step {best_step}", flush=True)
                    stop = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"model of step {best_step} (validation MAE {best:.4f}) is kept", flush=True)
    return step


def train_causal(model, data, args, device, rng, run_dir, log):
    """
    Causal code training: no code table. At every step each patient gets a random cut row k (labelled, with 3 rows of history
    or more); its code is fitted from 0 by `inner_steps` Adam steps on the history up to k alone, the labels counted only if their
    6-month window ended by the cut (the rule of the test); then the model learns to predict y at k from the state at k and
    that code (held fixed). So no code, and nothing the model learns from a code, ever saw the future of its patient.
    """
    train = data.train_idx
    cand = {}
    for i in train:
        r = np.flatnonzero(data.ym[data.off[i]:data.off[i + 1]])
        r = r[r >= 2]
        if len(r):
            cand[int(i)] = r
    pool = np.array(sorted(cand))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    start, step = time.time(), 0
    while step < args.steps and time.time() - start < args.seconds:
        perm = rng.permutation(pool)
        for a in range(0, len(perm) - args.batch + 1, args.batch):
            if step >= args.steps or time.time() - start >= args.seconds:
                break
            part = perm[a:a + args.batch]
            ks = [int(rng.choice(cand[int(i)])) for i in part]
            items_in = [(int(i), k + 1, data.months[data.off[i] + k] - 6.0) for i, k in zip(part, ks)]
            items_out = [(int(i), k + 1, None) for i, k in zip(part, ks)]
            warm = min(1.0, step / 500)
            c, _ = fit_codes(model, data, items_in, device, steps=args.inner_steps, lr=args.inner_lr, beta=warm)
            batch = data.pack(items_out, c, device, only_last=True)
            model.train()
            per, _ = model.loss(batch, beta=warm, per_trajectory=True)
            good = torch.isfinite(per)
            loss = torch.where(good, per, torch.zeros_like(per)).sum() / batch["mask"].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)):
                opt.step()
            sched.step()
            if step % 100 == 0:
                msg = (f"{step} t={time.time() - start:.0f}s loss {loss.item():.4f} finite {good.float().mean().item():.2f} "
                       f"|c| {c.norm(dim=1).mean().item():.3f}")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            step += 1
            if step % 1000 == 0:
                torch.save({"model": model.state_dict()}, f"{run_dir}/model_{step}.pt")
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--lr_code", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=LIGHT)
    ap.add_argument("--resume", help="a run folder: continue from its model.pt (weights and codes; a new Adam, no warmup)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate of the model weights")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait after each chunk of the evaluation")
    ap.add_argument("--eval_only", help="a run folder: score its model.pt (the training weight beta_max is used in the code fit)")
    ap.add_argument("--dim_z", type=int, default=CFG["dim_z"], help="latent size (the bottleneck)")
    ap.add_argument("--dec_hidden", type=int, nargs="*", default=CFG["decoder"]["hidden"], help="hidden widths of the decoder")
    ap.add_argument("--beta_max", type=float, default=1.0, help="weight of the SDE terms (drift residual, KL of the initial state); 0: no SDE")
    ap.add_argument("--no_code", action="store_true", help="no per-patient code; training on all labelled rows of each patient")
    ap.add_argument("--extra_features", action="store_true", help="append the causal time-series features of extra_features")
    ap.add_argument("--fc_weight", type=float, default=0.0, help="weight of the 12-month forecast loss of the drift (0: off); needs --no_code")
    ap.add_argument("--fc_cuts", type=int, default=4, help="cut visits per patient and step")
    ap.add_argument("--fc_samples", type=int, default=4, help="simulated paths per cut")
    ap.add_argument("--fc_sub", type=int, default=2, help="Euler sub-steps per month")
    ap.add_argument("--treatment_u", action="store_true", help="the treatment state of each visit is the forcing u(t) of the drift")
    ap.add_argument("--drop_engineered", action="store_true", help="remove the engineered score and visit summaries (ENGINEERED) from the inputs")
    ap.add_argument("--l1", action="store_true", help="train on the absolute error of the prediction (the challenge metric is MAE)")
    ap.add_argument("--real_only", action="store_true", help="train on the real labels only (one per patient, the last visit)")
    ap.add_argument("--real_flag", action="store_true", help="one more input: 1 at the row with the real label, 0 at the others")
    ap.add_argument("--weight_sqrt", action="store_true", help="derived labels weigh 1 / sqrt(derived labels of the patient), real labels 1")
    ap.add_argument("--val_frac", type=float, default=0.0, help="share of the training patients held out for a validation score (snapshots)")
    ap.add_argument("--val_every", type=int, default=1000, help="steps between validation scores")
    ap.add_argument("--patience", type=int, default=5, help="validation scores without improvement before the training stops")
    ap.add_argument("--causal", action="store_true", help="causal code training (train_causal): no code table")
    ap.add_argument("--inner_steps", type=int, default=4, help="Adam steps of the code fit (training and test, causal mode)")
    ap.add_argument("--inner_lr", type=float, default=0.15, help="learning rate of the code fit (causal mode)")
    ap.add_argument("--max_test", type=int, default=10 ** 9, help="test patients to score (a smoke test uses fewer)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = Data(args.data, extra=args.extra_features, treatment_u=args.treatment_u, drop_engineered=args.drop_engineered, real_flag=args.real_flag)
    data.use_w = args.weight_sqrt
    data.val_idx = np.array([], int)
    if args.val_frac:
        perm = rng.permutation(data.train_idx)
        k = int(args.val_frac * len(perm))
        data.val_idx, data.train_idx = np.sort(perm[:k]), np.sort(perm[k:])
    CFG["dim_feat"] = data.X.shape[1]
    if args.no_code:
        CFG["dim_mu"] = 0
    if args.treatment_u:
        CFG["dim_u"] = data.U.shape[1]
    CFG["dim_z"], CFG["decoder"] = args.dim_z, {**CFG["decoder"], "hidden": list(args.dec_hidden)}
    if args.eval_only:
        model = build_model(CFG, None, 1, None).to(device)
        model.load_state_dict(torch.load(f"{args.eval_only}/model.pt", map_location=device)["model"])
        ck = torch.load(f"{args.eval_only}/model.pt", map_location=device)
        res = evaluate(model, data, device, max_patients=args.max_test, beta=args.beta_max, pause=args.pause,
                       code_default=ck["codes"].mean(0))
        yaml.safe_dump(res, open(f"{args.eval_only}/metrics_refit2.yaml", "w"))
        return
    run_dir = time.strftime("runs/hs_seq_%Y%m%d-%H%M%S")
    os.makedirs(run_dir)
    np.save(f"{run_dir}/val_idx.npy", data.val_idx)
    yaml.safe_dump({"cfg": CFG, "args": vars(args)}, open(f"{run_dir}/config.yaml", "w"))
    if args.no_code:
        model = build_model(CFG, None, 1, None).to(device)
        model.l1 = args.l1
        print(f"run {run_dir}: no code, all labelled rows per step, {data.X.shape[1]} features, {len(data.train_idx)} train, "
              f"{len(data.test_idx)} test patients, device {device}", flush=True)
        t0, log = time.time(), open(f"{run_dir}/log.txt", "w")
        step = train_all_rows(model, data, args, device, rng, run_dir, log)
        torch.save({"model": model.state_dict(), "codes": torch.zeros(1, 0)}, f"{run_dir}/model.pt")
        print(f"trained {step} steps in {time.time() - t0:.0f}s", flush=True)
        from experiments.latent.eval_all_rows import score_all
        score_all(model, data, device, run_dir, pause=args.pause, dim_mu=CFG["dim_mu"])
        return
    if args.causal:
        model = build_model(CFG, None, 1, None).to(device)
        print(f"run {run_dir}: causal code training, {len(data.train_idx)} train, {len(data.test_idx)} test patients, "
              f"device {device}", flush=True)
        t0, log = time.time(), open(f"{run_dir}/log.txt", "w")
        step = train_causal(model, data, args, device, rng, run_dir, log)
        torch.save({"model": model.state_dict(), "codes": torch.zeros(1, CFG["dim_mu"])}, f"{run_dir}/model.pt")
        print(f"trained {step} steps in {time.time() - t0:.0f}s", flush=True)
        res = evaluate(model, data, device, max_patients=args.max_test, beta=1.0, pause=args.pause,
                       code_default=torch.zeros(CFG["dim_mu"]), fit_steps=args.inner_steps, fit_lr=args.inner_lr)
        yaml.safe_dump(res, open(f"{run_dir}/metrics.yaml", "w"))
        return
    train = data.train_idx
    index = {int(i): k for k, i in enumerate(train)}
    lengths = np.array([data.n_rows(i) for i in train])
    print(f"run {run_dir}: {len(train)} train, {len(data.test_idx)} test patients, device {device}", flush=True)

    model = build_model(CFG, None, 1, None).to(device)
    codes = torch.nn.Embedding(len(train), CFG["dim_mu"]).to(device)
    torch.nn.init.normal_(codes.weight, std=0.1)
    if args.resume:
        ck = torch.load(f"{args.resume}/model.pt", map_location=device)
        assert ck["train_ids"] == data.pid[train].tolist(), "the resumed run has other training patients"
        model.load_state_dict(ck["model"])
        codes.weight.data.copy_(ck["codes"].to(device))
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}, {"params": codes.parameters(), "lr": args.lr_code}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    n_warmup, start, step = (0 if args.resume else 500), time.time(), 0
    log = open(f"{run_dir}/log.txt", "w")
    while step < args.steps and time.time() - start < args.seconds:
        order = np.argsort(lengths + rng.uniform(0, 3, len(lengths)))  # similar lengths together, new groups each epoch
        parts = [order[a:a + args.batch] for a in range(0, len(order), args.batch)]
        for p in rng.permutation(len(parts)):
            if step >= args.steps or time.time() - start >= args.seconds:
                break
            part = parts[p]
            c = codes(torch.as_tensor(part, device=device))
            batch = data.pack([(int(train[k]), data.n_rows(train[k]), None) for k in part], c, device)
            warm = min(1.0, step / n_warmup) if n_warmup else 1.0
            model.train()
            loss, terms = model.loss(batch, beta=args.beta_max * warm)
            total = loss + warm * code_prior(c) / batch["mask"].sum().clamp(min=1)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            if step % 100 == 0:
                msg = f"{step} t={time.time() - start:.0f}s loss {loss.item():.4f} " + " ".join(
                    f"{k} {v.item():.4f}" for k, v in terms.items()) + f" |c| {codes.weight.norm(dim=1).mean().item():.3f}"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            step += 1
            if step % 5000 == 0:  # snapshots, to score the model along the training later
                torch.save({"model": model.state_dict(), "codes": codes.weight.detach().cpu(),
                            "train_ids": data.pid[train].tolist()}, f"{run_dir}/model_{step}.pt")
    torch.save({"model": model.state_dict(), "codes": codes.weight.detach().cpu(), "train_ids": data.pid[train].tolist()},
               f"{run_dir}/model.pt")
    print(f"trained {step} steps in {time.time() - start:.0f}s", flush=True)
    res = evaluate(model, data, device, max_patients=args.max_test, beta=args.beta_max, pause=args.pause,
                   code_default=codes.weight.detach().mean(0))
    yaml.safe_dump(res, open(f"{run_dir}/metrics.yaml", "w"))


if __name__ == "__main__":
    main()
