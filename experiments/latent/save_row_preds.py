"""
Save the prediction of a no-code model at every labelled row of the test patients (CPU, no training).
Row = index in the npz; the order is the order of data.test_idx. Used by shifted_validation.py.

    python -m experiments.latent.save_row_preds runs/hs_seq_<stamp>
"""
import argparse

import numpy as np
import torch

from experiments.latent import run_hs_seq as R
from experiments.latent.eval_all_rows import predict_rows
from experiments.latent.run import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--real_flag", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    data = R.Data(R.LIGHT, real_flag=args.real_flag)
    R.CFG["dim_feat"], R.CFG["dim_mu"] = data.X.shape[1], 0
    model = build_model(R.CFG, None, 1, None)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location="cpu")["model"])
    preds = predict_rows(model, data, data.test_idx, "cpu", dim_mu=0)
    rows, p = [], []
    for i, pr in zip(data.test_idx, preds):
        s = slice(data.off[i], data.off[i + 1])
        ok = np.flatnonzero(data.ym[s])
        rows.append(ok + data.off[i])
        p.append(pr[ok])
    np.savez(f"{args.run}/row_preds.npz", rows=np.concatenate(rows), pred=np.concatenate(p))
    print("saved", f"{args.run}/row_preds.npz", sum(len(r) for r in rows), "rows")


if __name__ == "__main__":
    main()
