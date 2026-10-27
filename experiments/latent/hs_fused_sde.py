"""
Fused-latent SDE model for the HS challenge. All train-folder patients; one held-out set (is_test_patient of hs_build.py, the same patients and
validation split as the reference runs in REFS: paired). Default input: raw (no hand-made history summaries, hs_e2e.RAW_DROP), with ethnicity.

At each step t of a patient (the steps of hs_build.py: visits, quarter ends, the cutoff, decision steps):
  1. observation: one linear projection (token) per modality: clinical features + mask flags, image embedding, RNAseq embedding (full width,
     256 / 128 values, standardized: the model learns its own compression; --emb_pca k gives k PCs instead). Cross-attention: the clinical token (always present) is the query; the keys / values are a learned null token and the
     image and RNAseq tokens (+ type embeddings); a missing modality is masked out of the keys. Residual (the clinical token stays whole) + layer
     norm, then one network of [result, presence flags] -> the fused observation (mean, logvar) in a 16-d latent space. With 3 tokens the
     attention is a patient-specific gate on the masked sum of the views in the H update of ALDSL. No product of experts: in run
     runs/hs_fused_sde_20261026-155353 the clinical expert took 97-99% of the fused precision.
  2. posterior: a forward GRU reads only the fused observations (mean, logvar) and the log gap, and gives q(z_t | data up to t) =
     N(m_t, diag exp(logv_t)) in the same 16-d space.
  3. prior: a latent SDE dz = f(z, u, s) dt + diag g(z, u, s) dW; u = the treatment flags in the interval (antibiotic, biologic, the 3 drugs,
     at the end of the interval, so that a start the day after a decision step counts), s = static covariates. Decoder: z -> the visit IHS4
     score (Laplace in log1p(score), so scores stay >= 0).
  4. forecast: a head on (m_t, logv_t) and the step kind -> the mean IHS4 of the next 6 months (median + Laplace scale; the labels and
     weights of head 1 of hs_e2e.py, the step kind gives the offset of the real label).
Loss = head (L1 + 0.1 Laplace NLL) + w_sde * (score NLL at the visits + beta * KL), with the filtering KL of a Kalman-type filter:
     KL(q_t || N(m_{t-1} + f(m_{t-1}) dt, v_{t-1} + g^2 dt))    (the Euler predict step of the SDE from the previous posterior; the Jacobian
                                                                  of f is left out of the variance), and KL(q_0 || N(z0_mean, exp(z0_logvar))).
--flare 1 (stage 2 of flare_model.py): the visit score = g(z) + F(t), F the excess of the flares (onsets = flare visits, a Poisson process
with a rate from z, the biologics, static and a patient code; amplitude per flare amortized from the GRU state; exponential decay), so that
z can stay slow; + the onset NLL and the amplitude KL in the loss; the rollout draws new flares.
Evaluation: hs_e2e.evaluate (head: held-out MAE, task 3 variants, quarter ends, causality check, test folders), the SDE rollout forecast
(Euler-Maruyama paths from q at the step on a 7-day grid, treatment held, the decoded median score averaged over the 6 months; the real label
gets a constant offset from training patients), and paired differences against the GRU and XGBoost.

The model is in components (experiments/latent/fused, built from experiments/latent/parts); this file is the command line and
re-exports the old names (make_model, prepare, extras, flare_terms, fit_codes, rollout_forecast, ...).

    ../environments/torch_gpu_env/Scripts/python.exe -m experiments.latent.hs_fused_sde [--seconds 600] [--smoke] [--device cpu]
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml

from experiments.latent.fused.batch import CONTROLS, ROLL_H, ROLL_J, ROLL_STEP, STATIC, STEP_DAYS, extras, prepare, roll_batch  # noqa: F401
from experiments.latent.fused.evaluate import REFS, evaluate_sde, flare_report, rollout_forecast  # noqa: F401
from experiments.latent.fused.inference import fit_codes  # noqa: F401
from experiments.latent.fused.model import FusedSDE, make_model, upgrade_state_dict  # noqa: F401
from experiments.latent.fused.objective import flare_terms, rollout_loss, sde_terms, score_nll  # noqa: F401
from experiments.latent.fused.train import train
from experiments.latent.hs_e2e import PATHS, Data, evaluate
from experiments.latent.parts.flares import REACH as FLARE_REACH  # noqa: F401
from experiments.latent.parts.layers import mlp, soft_clamp  # noqa: F401
from experiments.latent.parts.stats import boot  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dz", type=int, default=16, help="width of the latent space (observation, posterior, SDE)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--p_drop", type=float, default=0.5, help="probability to hide a modality of a patient in training")
    ap.add_argument("--emb_pca", type=int, default=0, help="0: the full embeddings (the model learns its own compression); k: k PCs")
    ap.add_argument("--emb_noise", type=float, default=0.3)
    ap.add_argument("--w_sde", type=float, default=1.0, help="weight of the SDE terms (score NLL + beta KL)")
    ap.add_argument("--w_pred", type=float, default=0.0, help="weight of the predictive score NLL (from the SDE predict step, before the visit)")
    ap.add_argument("--meas", type=int, default=0, help="1: measurement model (visit-type bias and scale; the rollout forecasts g(z) without them)")
    ap.add_argument("--l21", type=float, default=0.0, help="group-lasso strength on the embedding columns of the token maps (proximal step)")
    ap.add_argument("--beta", type=float, default=0.1, help="weight of the KL inside the SDE terms")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=5, help="stop after this many validation checks without improvement (hs_e2e.py: 5)")
    ap.add_argument("--pause", type=float, default=0.05)
    ap.add_argument("--train_pause", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="", help="features removed from the input, comma separated (ethnicity is kept by default)")
    ap.add_argument("--raw", type=int, default=1, help="1: no hand-made history summaries in the input (hs_e2e.RAW_DROP, RAW_INC)")
    ap.add_argument("--mod_events", type=int, default=0, help="1: an image / RNAseq sample enters the input once, at the first step that "
                    "sees it (0: at every step from its date on)")
    ap.add_argument("--w_recon", type=float, default=0.0, help="weight of the ALDSL view reconstruction: each modality token rebuilds its "
                    "full embedding (0: off)")
    ap.add_argument("--head_obs", type=int, default=0, help="1: the forecast head also reads the fused observation of the step (mean, "
                    "log-variance), a skip connection around the GRU and the KL-regularized posterior")
    ap.add_argument("--code_dim", type=int, default=0, help="size of the patient code (auto-decoder, prior N(0, I)) in the set point and "
                    "the diffusion of the SDE prior (needs --drift_mr 1); 0: off")
    ap.add_argument("--code_lr", type=float, default=0.01, help="learning rate of the patient codes (SparseAdam)")
    ap.add_argument("--code_struct", default="", help="structured code: sizes of the level, noise, speed, response blocks, e.g. 2,2,2,2 "
                    "(replaces --code_dim)")
    ap.add_argument("--code_var", type=int, default=0, help="1: variational codes (mean and log-sd per patient, KL to the prior)")
    ap.add_argument("--code_prior_rna", type=int, default=0, help="1: prior mean of the response block = f(RNAseq, presence, static)")
    ap.add_argument("--code_fit_steps", type=int, default=100, help="Adam steps to fit the code of a new patient (evaluation)")
    ap.add_argument("--onehot", type=int, default=0, help="1: ethnicity and smoking status as one-hot columns (hs_e2e.ONEHOT)")
    ap.add_argument("--drift_mr", type=int, default=0, help="1: mean-reverting drift -lam (z - mu(treatment, static)) + a bounded term")
    ap.add_argument("--sde_sub", type=float, default=0.0, help="Euler sub-steps of at most this many days in the training predict step "
                    "(7: the rollout grid; 0: one step over the gap)")
    ap.add_argument("--head_gru", type=int, default=0, help="1: the forecast head also reads the GRU state (the input of the posterior "
                    "layer), a skip connection around the KL-regularized posterior")
    ap.add_argument("--recon_latent", type=int, default=0, help="1: the posterior mean (the latent state, shared by the views) rebuilds both "
                    "embeddings, also the samples hidden by modality dropout (masked modelling)")
    ap.add_argument("--smoke", action="store_true", help="a short test: 1,500 training patients, 150 steps, 300 test patients")
    ap.add_argument("--resume", default="", help="run folder: continue the training of its model.pt (same transforms and split)")
    ap.add_argument("--start_step", type=int, default=-1, help="with --resume: the steps done so far (default: from model.pt)")
    ap.add_argument("--eval_only", default="", help="run folder: no training, evaluate_sde of its model.pt again (code fits, rollouts, "
                    "paired comparisons; a separate job after a cool-down)")
    ap.add_argument("--eval_quarter", type=int, default=1, help="with --eval_only: 0 skips the rollout at the quarter ends (5,000 rows)")
    ap.add_argument("--sde_eval", type=int, default=1, help="0: no evaluate_sde after the training (code fits and rollouts: GPU load; run "
                    "them later with --eval_only after a cool-down)")
    ap.add_argument("--flare", type=int, default=0, help="1: the flare part (onsets = flare visits, amplitude, decay) in the measurement "
                    "model of the SDE, with patient flare codes")
    ap.add_argument("--keep_last", type=int, default=0, help="1: keep the model of the last step, not the one of the best validation MAE")
    ap.add_argument("--w_roll", type=float, default=0.0, help="weight of the multi-step rollout loss (mean path of the SDE over 12 monthly "
                    "steps from the posterior mean at a random step, against the later scheduled visits; 0: off)")
    ap.add_argument("--flare_cut", type=float, default=0.0, help="with --flare: days after an onset in which the score NLLs train only the "
                    "flare part (the slow part and the scale without gradient: a cut of the feedback); 0: no cut")
    ap.add_argument("--device", default="", help="cpu or cuda (default: cuda if available)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        ck = torch.load(f"{args.eval_only}/model.pt", map_location=device, weights_only=False)
        data = Data(PATHS["train"], stats=ck["stats"])
        prepare(data)
        pool = np.random.default_rng(ck["args"]["seed"]).permutation(np.flatnonzero(~data.flag))
        k = int(0.1 * len(pool))
        data.val_idx, data.train_idx, data.test_idx = np.sort(pool[:k]), np.sort(pool[k:]), np.flatnonzero(data.flag)
        model = make_model(data, ck["args"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        evaluate_sde(model, data, device, args.eval_only, rollout=ck["args"]["w_sde"] > 0, fit_steps=args.code_fit_steps,
                     quarter=bool(args.eval_quarter))
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ck = torch.load(f"{args.resume}/model.pt", map_location=device) if args.resume else None
    if ck is not None:
        data = Data(PATHS["train"], stats=ck["stats"])
        start_step = args.start_step if args.start_step >= 0 else int(ck.get("step", 0))
    else:
        data = Data(PATHS["train"], emb_pca=args.emb_pca, drop=[c for c in args.drop.split(",") if c], raw=bool(args.raw),
                    mod_events=bool(args.mod_events), onehot=bool(args.onehot))
        start_step = 0
    prepare(data)
    pool = rng.permutation(np.flatnonzero(~data.flag))  # the validation split of hs_e2e.py (same seed)
    k = int(0.1 * len(pool))
    data.val_idx, data.train_idx, data.test_idx = np.sort(pool[:k]), np.sort(pool[k:]), np.flatnonzero(data.flag)
    if args.smoke:
        data.train_idx, data.val_idx = data.train_idx[:1500], data.val_idx[:200]
        args.steps, args.val_every = 150, 75
    run_dir = time.strftime("runs/hs_fused_sde_%Y%m%d-%H%M%S") + ("_smoke" if args.smoke else "")
    os.makedirs(run_dir)
    yaml.safe_dump({"args": vars(args)}, open(f"{run_dir}/config.yaml", "w"))
    model = make_model(data, vars(args)).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"])
    print(f"run {run_dir}: {len(data.train_idx)} train, {len(data.val_idx)} validation, {len(data.test_idx)} held-out patients, "
          f"{sum(p.numel() for p in model.parameters())} parameters, device {device}; resumed from {args.resume or 'none'} at step {start_step}", flush=True)
    t0, log = time.time(), open(f"{run_dir}/log.txt", "w")
    step, best_step, best = train(model, data, args, device, log, start_step)
    torch.save({"model": model.state_dict(), "stats": data.stats, "args": vars(args), "step": step, "best_step": best_step},
               f"{run_dir}/model.pt")
    print(f"trained {step} steps in {time.time() - t0:.0f}s", flush=True)
    if model.flare:
        lines = flare_report(model, data)
        print("\n".join(lines), flush=True)
        with open(f"{run_dir}/flare.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    mx = 300 if args.smoke else None
    evaluate(model, data, device, run_dir, pause=args.pause, max_patients=mx)
    if args.sde_eval:
        evaluate_sde(model, data, device, run_dir, max_patients=mx, rollout=args.w_sde > 0, fit_steps=args.code_fit_steps)


if __name__ == "__main__":
    main()
