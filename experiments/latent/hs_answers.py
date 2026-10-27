"""
The answer files of the challenge from the end-to-end model (hs_e2e.py): test_1 and test_2 together, one file per exercise.

  exercise1                 all patients                     head 1 at the final step (the cutoff), all modalities
  exercise2                 the patients of its template     head 1, and super_responder_prob (hs_task2.py: head 1 against the untreated
                                                             forecast of head 2 from the decision step, on the scale of the real label)
  exercise3_images          the patients with an image       head 1 with the RNAseq hidden (the clinical + image model)
  exercise3_images_rnaseq   the patients with image + RNA    head 1 with all modalities (the clinical + image + RNAseq model)
The patient lists are the templates of the two test folders. Every listed patient gets a value (a patient of the exercise 2 list
without an open line in the data gets the median probability; the count is printed). Predictions below 0 are set to 0.

    python -m experiments.latent.hs_answers runs/hs_e2e_<stamp> [--team teamName] [--out runs/answers] [--blend runs/xgb_e2e/test_preds.npz]
"""
import argparse
import csv
import os

import numpy as np
import torch

from experiments.latent.hs_e2e import PATHS, Data, predict
from experiments.latent.hs_task2 import counterfactual, load, open_lines, scale_factor, sr_prob

TEMPLATE = "test_data/hs_public_dataset/public/{folder}/answers_teamName_{ex}.csv"
EXERCISES = ["exercise1", "exercise2", "exercise3_images", "exercise3_images_rnaseq"]


def read_ids(path):
    with open(path, newline="") as f:
        return [r["patient_id"] for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--team", default="teamName")
    ap.add_argument("--out", default="runs/answers")
    ap.add_argument("--blend", help="test_preds.npz of xgb_e2e.py --tests: average the forecasts with XGBoost")
    ap.add_argument("--w_gru", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train, ck = load(args.run, device)
    a, _ = scale_factor(model, train, list(np.flatnonzero(~train.flag)), device)
    del train
    preds = {}
    for folder in ("test_1", "test_2"):
        data = Data(PATHS[folder], stats=ck["stats"])
        pats = list(range(data.n_pat))
        full = predict(model, data, pats, device, pause=0.05)[0]
        image = predict(model, data, pats, device, hide_rna=True, pause=0.05)[0]
        lines = open_lines(data, pats)
        cf = counterfactual(model, data, lines, device)
        prob = dict(zip([l[0] for l in lines], sr_prob(cf["y1"], cf["b1"], a * cf["y0"])))
        for i in pats:
            preds[str(data.pid[i])] = {"full": max(float(full[i]), 0.0), "image": max(float(image[i]), 0.0), "prob": prob.get(i, np.nan)}
        print(f"{folder}: {data.n_pat} patients predicted, {len(lines)} with an open biologic line and its decision step", flush=True)
    if args.blend:  # the mean with XGBoost (xgb_e2e.py --tests): w_gru * GRU + (1 - w_gru) * XGBoost; the probability stays the GRU's
        b = np.load(args.blend)
        for folder in ("test_1", "test_2"):
            for p, f_, im in zip(b[f"{folder}_patient_id"], b[f"{folder}_full"], b[f"{folder}_image"]):
                v = preds[str(p)]
                v["full"] = max(args.w_gru * v["full"] + (1 - args.w_gru) * float(f_), 0.0)
                v["image"] = max(args.w_gru * v["image"] + (1 - args.w_gru) * float(im), 0.0)
        print(f"blended with {args.blend}: {args.w_gru:.2f} GRU + {1 - args.w_gru:.2f} XGBoost")
    known = [v["prob"] for v in preds.values() if not np.isnan(v["prob"])]
    fallback = float(np.median(known))
    for ex in EXERCISES:
        ids = read_ids(TEMPLATE.format(folder="test_1", ex=ex)) + read_ids(TEMPLATE.format(folder="test_2", ex=ex))
        missing = [p for p in ids if p not in preds]
        assert not missing, f"{ex}: {len(missing)} listed patients have no prediction"
        path = f"{args.out}/answers_{args.team}_{ex}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            if ex == "exercise2":
                w.writerow(["patient_id", "pred_avg_ihs4_next6", "super_responder_prob"])
                n_fb = 0
                for p in ids:
                    pr = preds[p]["prob"]
                    if np.isnan(pr):
                        pr, n_fb = fallback, n_fb + 1
                    w.writerow([p, f"{preds[p]['full']:.4f}", f"{pr:.4f}"])
                print(f"{path}: {len(ids)} patients ({n_fb} without an open line in the data get the median probability {fallback:.3f})")
            else:
                key = "image" if ex == "exercise3_images" else "full"
                w.writerow(["patient_id", "pred_avg_ihs4_next6"])
                for p in ids:
                    w.writerow([p, f"{preds[p][key]:.4f}"])
                print(f"{path}: {len(ids)} patients")


if __name__ == "__main__":
    main()
