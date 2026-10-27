"""
MAE, RMSE and R2 (original units) of the SDE and of simple baselines on the validation patients of a run_severity run:
the last H values of each patient are held out. Baselines: train mean, last value, mean of the history over the last
`window` raw time units (6, as the "6-month mean" if time is in months).

    python -m experiments.latent.compare_baselines runs/severity_code_<stamp> [window]
"""
import sys

import numpy as np
import torch

from experiments.latent.run import build_model
from experiments.latent.run_severity import CFG, T_SCALE, fit_codes, forecast, load


def metrics(pred, true):
    e = pred - true
    return {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
            "R2": float(1 - (e ** 2).sum() / ((true - true.mean()) ** 2).sum())}


def main(run_dir, window=6.0, H=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{run_dir}/model.pt", map_location="cpu")
    series, mean, std = load()
    mean, std = float(mean), float(std)
    model = build_model(CFG, None, 1, None)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    val = [p for p in ck["val_ids"] if len(series[p][0]) >= H + 3]
    hist = [(series[p][0][:-H], series[p][1][:-H]) for p in val]
    fut_t = np.stack([series[p][0][-H:] for p in val])
    true = np.stack([series[p][1][-H:] for p in val]) * std + mean
    c = torch.cat([fit_codes(model, hist[i:i + 200], CFG["dim_mu"], device) for i in range(0, len(hist), 200)])
    pred = torch.cat([forecast(model, hist[i:i + 200], fut_t[i:i + 200], c[i:i + 200], device)
                      for i in range(0, len(hist), 200)]).cpu().numpy() * std + mean
    last = np.array([h[1][-1] for h in hist])[:, None] * std + mean
    recent = np.array([h[1][h[0] >= h[0][-1] - window / T_SCALE].mean() for h in hist])[:, None] * std + mean
    train_mean = np.mean(np.concatenate([series[p][1] for p in ck["train_ids"]])) * std + mean
    rows = {"Train mean": np.full_like(true, train_mean), "Persistence: last value": np.broadcast_to(last, true.shape),
            f"Persistence: {window:g}-unit mean": np.broadcast_to(recent, true.shape), "Latent SDE": pred}
    print(f"{len(val)} validation patients, last {H} values each ({true.size} rows)")
    for name, p in rows.items():
        m = metrics(p, true)
        print(f"{name:32s} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
