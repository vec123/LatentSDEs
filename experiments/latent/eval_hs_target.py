"""
Scores a run_severity model on the official target of hs_sequences_train.npz: y(t) = mean IHS4 score in (t, t + 6 months].
For a test patient (is_test_patient) and a step t, the history is the CSV scores up to t, the code is fitted on it (model
frozen), the drift rolls out from the last history time and the prediction is the mean of the decoded rollout at
t + 0.5 ... t + 6 months. Two kinds of step, scored apart: the last step (y_is_real = 1, the official rows) and one random
derived step per patient (y_is_real = 0). Baselines: train mean of y, last score, mean of the scores of the past 6 months.

    python -m experiments.latent.eval_hs_target runs/severity_code_<stamp> [max_patients] [only_val]
"""
import sys

import numpy as np
import torch

from experiments.latent.compare_baselines import metrics
from experiments.latent.run import build_model
from experiments.latent.run_severity import CFG, T_SCALE, fit_codes, forecast, load

# the arrays of hs_sequences_train.npz this needs, with the patient ids as str (the file has numpy 2 pickles, numpy 1 cannot read it)
NPZ = "test_data/hs_public_dataset/hs_sequences_light.npz"


def main(run_dir, max_patients=10 ** 9, only_val=True, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    ck = torch.load(f"{run_dir}/model.pt", map_location="cpu")
    series, mean, std = load()
    mean, std = float(mean), float(std)
    d = np.load(NPZ)
    off, pid, flag = d["offsets"], d["patient_id"], d["is_test_patient"].astype(bool)
    t_row, y, ym, yr = d["time_months"], d["y"], d["y_mask"].astype(bool), d["y_is_real"].astype(bool)
    allowed = set(ck["val_ids"]) if only_val else None  # patients the model never trained on
    train_y = np.concatenate([y[off[i]:off[i + 1]][ym[off[i]:off[i + 1]]] for i in range(len(pid)) if not flag[i]])
    rows = {"real": [], "derived": []}
    for i in np.flatnonzero(flag)[:]:
        p = pid[i]
        if p not in series or (allowed is not None and p not in allowed):
            continue
        s = slice(off[i], off[i + 1])
        last = np.flatnonzero(yr[s] & ym[s])
        der = np.flatnonzero(ym[s] & ~yr[s] & (t_row[s] > 6))
        if len(last):
            rows["real"].append((p, t_row[s][last[0]], y[s][last[0]]))
        if len(der):
            k = rng.choice(der)
            rows["derived"].append((p, t_row[s][k], y[s][k]))
    model = build_model(CFG, None, 1, None)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    grid = np.arange(1, 13) * 0.5
    for kind, items in rows.items():
        items = [(p, t, v) for p, t, v in items if (series[p][0] * T_SCALE <= t + 0.02).sum() >= 3][:max_patients]
        hist = []
        for p, t, v in items:
            keep = series[p][0] * T_SCALE <= t + 0.02
            hist.append((series[p][0][keep], series[p][1][keep]))
        fut = np.stack([(t + grid) / T_SCALE for _, t, _ in items])
        true = np.array([v for _, _, v in items])
        c = torch.cat([fit_codes(model, hist[i:i + 200], CFG["dim_mu"], device) for i in range(0, len(hist), 200)])
        pred = torch.cat([forecast(model, hist[i:i + 200], fut[i:i + 200], c[i:i + 200], device)
                          for i in range(0, len(hist), 200)]).cpu().numpy().mean(1) * std + mean
        last_s = np.array([h[1][-1] for h in hist]) * std + mean
        past = np.array([h[1][h[0] > h[0][-1] - 6 / T_SCALE].mean() for h in hist]) * std + mean
        print(f"\n{kind} steps: {len(items)} patients, y mean {true.mean():.2f} std {true.std():.2f}")
        for name, p in (("Train mean", np.full_like(true, train_y.mean())), ("Persistence: last score", last_s),
                        ("Persistence: past 6-month mean", past), ("Latent SDE", pred)):
            m = metrics(p, true)
            print(f"{name:32s} RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}  R2 {m['R2']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9,
         (sys.argv[3] != "0") if len(sys.argv) > 3 else True)
