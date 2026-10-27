"""
Plots of a finished run_severity run, from its model.pt (no training):
  forecast_patients.png  predicted vs true for 8 validation patients (codes fitted on the history only, CPU)
  codes_pca.png          PCA of the learned codes of the training patients, one point per patient

    python -m experiments.latent.plot_severity runs/severity_code_<stamp>
"""
import sys

import numpy as np
import torch
import yaml

from experiments.latent.run import build_model
from experiments.latent.run_severity import CFG, T_SCALE, fit_codes, load, plot_patients


def main(run_dir, H=3):
    device = "cpu"
    ck = torch.load(f"{run_dir}/model.pt", map_location="cpu")
    series, mean, std = load()
    mean, std = float(mean), float(std)
    model = build_model(CFG, None, 1, None)
    model.load_state_dict(ck["model"])
    model.eval()

    val = [p for p in ck["val_ids"] if len(series[p][0]) >= H + 3]
    hist = [(series[p][0][:-H], series[p][1][:-H]) for p in val]
    fut_t = np.stack([series[p][0][-H:] for p in val])
    fut_v = torch.as_tensor(np.stack([series[p][1][-H:] for p in val]), dtype=torch.float32)
    pick = np.argsort([-len(h[0]) for h in hist])[:60:3]  # the 20 patients plotted
    sub = lambda a: [a[i] for i in pick] if isinstance(a, list) else a[pick]
    c = fit_codes(model, sub(hist), CFG["dim_mu"], device)
    plot_patients(model, series, sub(val), sub(hist), sub(fut_t), sub(fut_v), c, mean, std, device,
                  f"{run_dir}/forecast_patients.png", n=len(pick))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = ck["codes"].numpy()
    ids = ck["train_ids"]
    centered = codes - codes.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ vt[:2].T
    var = s ** 2 / (s ** 2).sum()
    level = np.array([series[p][1].mean() * std + mean for p in ids])
    slope = np.array([np.polyfit(series[p][0] * T_SCALE, series[p][1] * std, 1)[0] if len(series[p][0]) > 2 else 0.0
                      for p in ids])
    length = np.array([len(series[p][0]) for p in ids])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, v, name, cmap in ((axes[0], level, "mean severity", "viridis"),
                              (axes[1], np.clip(slope, *np.percentile(slope, [2, 98])), "slope per time unit", "coolwarm"),
                              (axes[2], np.log(length), "log number of visits", "magma")):
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=v, s=3, alpha=0.6, cmap=cmap, rasterized=True)
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel(f"PC1 ({var[0]:.0%})")
        ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    fig.suptitle(f"PCA of the {len(ids)} learned patient codes (first 2 PCs; variance share of all 8: "
                 + ", ".join(f"{v:.0%}" for v in var) + ")")
    fig.tight_layout()
    fig.savefig(f"{run_dir}/codes_pca.png", dpi=120)
    print("explained variance", np.round(var, 3), "corr PC1/level", np.corrcoef(pc[:, 0], level)[0, 1],
          "PC2/level", np.corrcoef(pc[:, 1], level)[0, 1])


if __name__ == "__main__":
    main(sys.argv[1])
