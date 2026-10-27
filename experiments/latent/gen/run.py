"""
Fit the generator and run the 6-month test.

    python -m experiments.latent.gen.run [--patients N] [--steps K] [--batch B] [--device cpu|cuda] [--threads T] [--pause S]
                                         [--test 0|1] [--out runs/gen]

--batch 0: the full batch. On the laptop GPU use mini-batches (and/or --pause): full-batch steps throttled it within ~20 s.
"""
import argparse
import os

import torch

from experiments.latent.gen.data import load, to
from experiments.latent.gen.evaluate import six_month_test
from experiments.latent.gen.fit import fit
from experiments.latent.gen.model import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=0, help="a random subset (0: all)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=0, help="patients per step (0: all)")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds of rest after each step (GPU duty cycle)")
    ap.add_argument("--test", type=int, default=1, help="0: no 6-month test (timing runs)")
    ap.add_argument("--out", default="runs/gen")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    rec = to(load(a.patients or None), a.device)
    N = len(rec["pats"])
    log(f"{N} patients ({int(rec['held'].sum())} held-out), {len(rec['v_y'])} scored visits ({float(rec['v_win'].float().mean()):.0%} in the "
        f"cut window), {len(rec['f_t'])} onsets, {len(rec['k_init'])} knots; device {a.device}, batch {a.batch or N}, {a.steps} steps")
    model = Generator(rec).to(a.device)
    fit(model, rec, a.steps, a.batch or N, a.lr, a.pause, log)
    torch.save(model.state_dict(), f"{a.out}/model.pt")
    if a.test:
        six_month_test(model, rec, log, out=f"{a.out}/six_month.npz")
    with open(f"{a.out}/report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
