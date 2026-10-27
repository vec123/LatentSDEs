"""
Auto-decoder latent SDE on the 1D HS severity series (test_data/hs_public_dataset/public/hs_severity_evolution_train.csv).

Each patient has a learned code c_i (a row of an embedding table, N(0, I) prior) that goes in as mu: the forward GRU posterior,
the drift f(z, c_i) and the diffusion read it. There is no encoder for c_i. The model and all codes train together. For a new
patient, the model is frozen and only the code is fitted on the history (fit_codes). The forecast test holds out the last
`horizon` values of each validation patient, fits the codes on the rest, rolls out the drift (no noise) from the posterior mean
at the last observed time, and compares with persistence (the last value).

    python -m experiments.latent.run_severity [--seconds 165] [--steps 4000]

Run it with environments/torch_gpu_env (CUDA). Time is divided by 30 (the mean time is about 30), the value is standardized.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.latent.run import build_model
from src.latent.batch import make_batch

CSV = "test_data/hs_public_dataset/public/hs_severity_evolution_train.csv"
T_SCALE = 30.0
CFG = {
    "dim_z": 4, "dim_mu": 8, "dim_u": 0,
    "encoder": {"type": "none"},
    "decoder": {"type": "gaussian", "hidden": [32], "std_init": 0.5, "reads_mu": False},
    "posterior": {"type": "sequence", "hidden": 64, "bidirectional": False, "forward_only": True},
    "drift": {"hidden": [64, 64]},
    "diffusion": {"type": "constant"},
    "g_init": 0.3, "n_samples": 4, "n_quad": 32,
}


def load(path=CSV):
    d = pd.read_csv(path).sort_values(["patient_id", "time"])
    mean, std = d["value"].mean(), d["value"].std()
    series = {}
    for pid, g in d.groupby("patient_id"):
        t = g["time"].to_numpy() / T_SCALE + 1e-4 * np.arange(len(g))  # strictly increasing
        series[pid] = (t, (g["value"].to_numpy() - mean) / std)
    return series, mean, std


def pad(items, codes, device):
    """items: list of (t, v); codes (B, P). One batch dict, padded with mask 0 and the last time repeated."""
    B, N = len(items), max(2, max(len(t) for t, _ in items))
    t, x, m = np.zeros((B, N)), np.zeros((B, N, 1)), np.zeros((B, N, 1))
    for b, (tb, vb) in enumerate(items):
        n = len(tb)
        t[b, :n], t[b, n:], x[b, :n, 0], m[b, :n, 0] = tb, tb[-1], vb, 1.0
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return make_batch(f(t), f(x), f(m), codes)


def chunks(order, size, rng):
    """Batches of patients of similar length (sorted), in random order."""
    parts = [order[i:i + size] for i in range(0, len(order), size)]
    return [parts[i] for i in rng.permutation(len(parts))]


def forecast_nll(model, batch, c, F=3, S=2, K=8):
    """
    Multi-step prediction term. A random cut k per patient: the GRU posterior reads only the steps up to k, the prior SDE
    (drift + noise, Euler-Maruyama, K sub-steps per gap) runs from z(t_k) through the next F observations, and the decoder
    scores them. Only the drift and the code can predict them: the GRU has no data after the cut. Mean negative log
    likelihood per scored value; patients with fewer than 3 values do not count.
    """
    t, x, mask = batch["t"], batch["x"], batch["mask"]
    B, N = t.shape
    n = mask[..., 0].sum(1).long()
    k = torch.minimum((torch.rand(B, device=t.device) * (n - 2).clamp(min=1)).long() + 1, (n - 2).clamp(min=1))
    keep = (torch.arange(N, device=t.device)[None] <= k[:, None]).unsqueeze(-1).to(mask.dtype)
    model.posterior.condition({**batch, "x": x * mask * keep, "mask": mask * keep})
    t_cut = t.gather(1, k[:, None])
    m, logv, _, _ = model.posterior(t_cut)
    z = (m[:, 0] + (0.5 * logv[:, 0]).exp() * torch.randn(S, B, m.shape[-1], device=t.device)).reshape(S * B, -1)
    cc, t_now, u = c.repeat(S, 1), t_cut[:, 0].repeat(S), c.new_zeros(S * B, 0)
    total, count = 0.0, 0.0
    for j in range(1, F + 1):
        valid = ((k + j) < n) & (n >= 3)
        idx = (k + j).clamp(max=N - 1)[:, None]
        T, xj = t.gather(1, idx)[:, 0].repeat(S), x[:, :, 0].gather(1, idx).repeat(S, 1)
        h = (T - t_now) / K
        for _ in range(K):
            G = model.diffusion(z, cc, t_now)
            z = z + model.drift(z, cc, u) * h[:, None] + G * h.clamp(min=0).sqrt()[:, None] * torch.randn_like(z)
            t_now = t_now + h
        lp = model.decoder(z, cc).log_prob(xj).view(S, B).mean(0)
        total, count = total - (lp * valid).sum(), count + valid.sum()
    return total / count.clamp(min=1)


def code_prior(c):
    return 0.5 * c.pow(2).sum()


def fit_codes(model, items, dim_c, device, steps=80, lr=0.05):
    """Codes for new patients: the model is frozen, only the codes train (Adam), N(0, I) prior. Returns (B, P)."""
    for p in model.parameters():
        p.requires_grad_(False)
    c = torch.zeros(len(items), dim_c, device=device, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    model.train()  # cuDNN GRU backward needs train mode; the model has no dropout
    for _ in range(steps):
        batch = pad(items, c, device)
        loss, _ = model.loss(batch)
        n = batch["mask"].sum()
        (loss + code_prior(c) / n).backward()
        opt.step()
        opt.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)
    return c.detach()


@torch.no_grad()
def forecast(model, hist, future_t, c, device, dt=0.02):
    """Drift rollout (no noise) from the posterior mean at each history's last time to the future times; decoded means (B, H)."""
    model.eval()
    batch = pad(hist, c, device)
    model.posterior.condition({**batch, "x": batch["x"] * batch["mask"], "mask": batch["mask"]})
    last = torch.tensor([h[0][-1] for h in hist], dtype=torch.float32, device=device)
    z = model.posterior(last[:, None])[0][:, 0]
    out, t_now = [], last.clone()
    T = torch.as_tensor(future_t, dtype=torch.float32, device=device)  # (B, H)
    u = torch.zeros(len(hist), 0, device=device)
    for k in range(T.shape[1]):
        while (t_now < T[:, k]).any():
            step = torch.clamp(torch.minimum(T[:, k] - t_now, torch.full_like(t_now, dt)), min=0)
            z = z + model.drift(z, c, u) * step[:, None]
            t_now = t_now + step
        out.append(model.decoder(z, c).mean[:, 0])
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=165.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--lr_code", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--official_split", action="store_true",
                    help="validation = the is_test_patient patients of hs_sequences_train.npz (never trained on)")
    ap.add_argument("--skip_eval", action="store_true", help="no last-3-values forecast test (use eval_hs_target)")
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the multi-step forecast term (0: off)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    series, mean, std = load()
    ids = np.array(sorted(series))
    perm = rng.permutation(len(ids))
    val_ids = set(ids[perm[:len(ids) // 10]])
    if args.official_split:  # light copy of hs_sequences_train.npz, see experiments/latent/eval_hs_target.py
        d = np.load("test_data/hs_public_dataset/hs_sequences_light.npz")
        val_ids = set(d["patient_id"][d["is_test_patient"].astype(bool)]) & set(ids)
    train_ids = [p for p in ids if p not in val_ids]
    index = {p: i for i, p in enumerate(train_ids)}
    order = sorted(train_ids, key=lambda p: len(series[p][0]))
    run_dir = time.strftime("runs/severity_code_%Y%m%d-%H%M%S")
    os.makedirs(run_dir)
    yaml.safe_dump({"cfg": CFG, "args": vars(args), "mean": float(mean), "std": float(std)}, open(f"{run_dir}/config.yaml", "w"))
    print(f"run {run_dir}: {len(train_ids)} train, {len(val_ids)} val patients, device {device}", flush=True)

    model = build_model(CFG, None, 1, None).to(device)
    codes = torch.nn.Embedding(len(train_ids), CFG["dim_mu"]).to(device)
    torch.nn.init.normal_(codes.weight, std=0.1)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3}, {"params": codes.parameters(), "lr": args.lr_code}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    n_warmup, start, step = 500, time.time(), 0
    log = open(f"{run_dir}/log.txt", "w")
    while step < args.steps and time.time() - start < args.seconds:
        for part in chunks(order, args.batch, rng):
            if step >= args.steps or time.time() - start >= args.seconds:
                break
            idx = torch.tensor([index[p] for p in part], device=device)
            c = codes(idx)
            batch = pad([series[p] for p in part], c, device)
            beta = min(1.0, step / n_warmup)
            model.train()
            loss, terms = model.loss(batch, beta=beta)
            n = batch["mask"].sum()
            total = loss + beta * code_prior(c) / n
            if args.lam:
                fc = forecast_nll(model, batch, c)
                total = total + args.lam * beta * fc
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            if step % 100 == 0:
                msg = f"{step} t={time.time() - start:.0f}s loss {loss.item():.4f} " + " ".join(
                    f"{k} {v.item():.4f}" for k, v in terms.items()) + f" |c| {codes.weight.norm(dim=1).mean().item():.3f}" + (f" fc {fc.item():.4f}" if args.lam else "")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            step += 1
    torch.save({"model": model.state_dict(), "codes": codes.weight.detach().cpu(), "train_ids": train_ids,
                "val_ids": sorted(val_ids), "mean": mean, "std": std}, f"{run_dir}/model.pt")
    print(f"trained {step} steps in {time.time() - start:.0f}s", flush=True)

    if args.skip_eval:
        return
    # forecast test on the validation patients: hold out the last `horizon` values, fit codes on the rest
    H = args.horizon
    val = [p for p in sorted(val_ids) if len(series[p][0]) >= H + 3]
    hist = [(series[p][0][:-H], series[p][1][:-H]) for p in val]
    fut_t = np.stack([series[p][0][-H:] for p in val])
    fut_v = torch.as_tensor(np.stack([series[p][1][-H:] for p in val]), dtype=torch.float32, device=device)
    c = torch.cat([fit_codes(model, hist[i:i + 200], CFG["dim_mu"], device) for i in range(0, len(hist), 200)])
    pred = torch.cat([forecast(model, hist[i:i + 200], fut_t[i:i + 200], c[i:i + 200], device) for i in range(0, len(hist), 200)])
    last = torch.as_tensor([h[1][-1] for h in hist], dtype=torch.float32, device=device)[:, None]
    std = float(std)
    mae = lambda a: (a - fut_v).abs().mean().item() * std
    res = {"n_val_patients": len(val), "mae_sde": mae(pred), "mae_persistence": mae(last.expand_as(fut_v)),
           "mae_per_step_sde": ((pred - fut_v).abs().mean(0) * std).tolist(),
           "mae_per_step_persistence": ((last - fut_v).abs().mean(0) * std).tolist()}
    print(res, flush=True)
    yaml.safe_dump(res, open(f"{run_dir}/forecast.yaml", "w"))
    plot_patients(model, series, val, hist, fut_t, fut_v, c, mean, std, device, f"{run_dir}/forecast_patients.png")


def plot_patients(model, series, val, hist, fut_t, fut_v, c, mean, std, device, path, n=20):
    """Predicted against true for n validation patients (4 x 5 for 20) (the longest histories): the history, the held-out true values,
    and the drift rollout (dense, no noise) from the last history time through the held-out times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = np.argsort([-len(h[0]) for h in hist])[:n * 3:3] if len(hist) > n else np.arange(len(hist))  # spread over long histories
    last = np.array([hist[i][0][-1] for i in pick])
    grid = last[:, None] + (fut_t[pick, -1] - last)[:, None] * np.linspace(0, 1, 40)[None]
    with torch.no_grad():
        curve = forecast(model, [hist[i] for i in pick], grid, c[pick], device).cpu().numpy() * std + mean
        point = forecast(model, [hist[i] for i in pick], fut_t[pick], c[pick], device).cpu().numpy() * std + mean
    cols = 5 if n > 8 else 4
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(3.6 * cols, 3.0 * -(-n // cols)), sharey=False)
    for ax, k, i in zip(axes.ravel(), range(n), pick):
        ht, hv = hist[i][0] * T_SCALE, hist[i][1] * std + mean
        ft, fv = fut_t[i] * T_SCALE, fut_v[i].cpu().numpy() * std + mean
        ax.plot(ht, hv, "o-", color="#6b6b66", ms=3, lw=1, label="history")
        ax.plot(ft, fv, "o", color="#222220", ms=6, label="true (held out)")
        ax.plot(grid[k] * T_SCALE, curve[k], "-", color="#2a78d6", lw=2, label="SDE rollout")
        ax.plot(ft, point[k], "s", color="#eb6834", ms=6, label="SDE at held-out times")
        ax.axhline(hv[-1], color="#6b6b66", ls=":", lw=1, label="persistence")
        ax.set_title(f"{val[i]}", fontsize=9)
        ax.set_xlabel("time")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle("Severity forecast: predicted vs true, validation patients (codes fitted on the history only)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
