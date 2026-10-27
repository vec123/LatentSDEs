"""
The training loop: batches of patients of similar length; loss = head (L1 + 0.1 Laplace NLL) + w_sde (score NLL + onset NLL + beta KL) +
w_pred predictive NLL [+ w_roll rollout loss] [+ the code priors per step] [+ w_recon view reconstruction]; the dense parameters with Adam
(exponential decay), the patient-code rows with SparseAdam; the l2,1 proximal step on the token maps; early stopping on the validation MAE
of the real labels (or keep_last).
"""
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.latent.fused.batch import extras, roll_batch
from experiments.latent.fused.objective import rollout_loss, sde_terms
from experiments.latent.hs_e2e import predict
from experiments.latent.parts.losses import laplace_terms, view_recon
from experiments.latent.parts.regularizers import group_prox, n_used


def token_maps(model):
    """the view token maps (image, RNAseq): the groups of the l2,1 penalty"""
    return [model.t_img.weight, model.t_rna.weight]


def log_line(model, args, step, t, l1, rec, kl, pred, rate, roll, ri, rr):
    fl = model.flare
    return (f"{step} t={t:.0f}s head L1 {l1.item():.4f} score NLL {rec.item():.4f} KL/step {kl.item():.4f} predictive NLL {pred.item():.4f}"
            + (f" rollout NLL {roll.item():.4f}" if args.w_roll > 0 else "")
            + (f" onset NLL/step {rate.item():.4f} flare: half-life {np.log(2) * fl.dur.bias.exp().item():.0f}d sd log A "
               f"{fl.ls_A.exp().item():.2f} patient sd {' '.join(f'{x:.2f}' for x in fl.ls_c.exp().tolist())}"
               + (f" cut share {model.last_cut_share:.2f}" if fl.cut else "") if fl is not None else "")
            + (" dims in use (image, RNAseq) %s" % n_used(token_maps(model)) if args.l21 > 0 else "")
            + (f" reconstruction MSE (image, RNAseq) {ri.item():.3f} {rr.item():.3f}" if args.w_recon > 0 else ""))


def train(model, data, args, device, log, start_step=0):
    """start_step > 0 (a resumed run): the learning rate continues its decay from there (Adam starts fresh), and the loaded model is the
    first best state"""
    rng = np.random.default_rng(args.seed + start_step)
    tr, val = data.train_idx, data.val_idx
    lengths = np.array([data.n_rows(i) for i in tr])
    yv = np.array([data.y[data.final_row(i)] for i in val])
    sparse = [m.weight for m in model.modules() if isinstance(m, nn.Embedding)]  # the patient-code rows
    main = [p for p in model.parameters() if all(p is not q for q in sparse)]
    opt = torch.optim.Adam(main, lr=args.lr * 0.9 ** (start_step / 1000))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9 ** (1 / 1000))
    code_opt = torch.optim.SparseAdam(sparse, lr=args.code_lr) if sparse else None  # rows outside the batch do not move
    start, step, best, best_step, best_state, stop = time.time(), start_step, float("inf"), start_step, None, False

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    if start_step:
        best = float(np.mean(np.abs(predict(model, data, list(val), device)[0] - yv)))
        best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        say(f"{step} resumed: validation MAE (real labels) {best:.4f}, learning rate {opt.param_groups[0]['lr']:.2e}")

    while step < start_step + args.steps and time.time() - start < args.seconds and not stop:
        order = np.argsort(lengths + rng.uniform(0, 3, len(lengths)))
        parts = [order[a:a + args.batch] for a in range(0, len(order), args.batch)]
        for p in rng.permutation(len(parts)):
            if step >= start_step + args.steps or time.time() - start >= args.seconds:
                break
            pats = [int(tr[k]) for k in parts[p]]
            batch = data.pack(pats, device, drop=args.p_drop, rng=rng, noise=args.emb_noise)
            ex = extras(data, pats, device)
            mu, b = model.forecast(model.encode(batch))
            l1, nll = laplace_terms(batch["y"], mu, b, batch["w"])
            loss = l1 + 0.1 * nll
            rec = kl = pred = rate = roll = torch.zeros(())
            if args.w_sde > 0 or args.w_pred > 0:  # both 0: no SDE terms at all (the forecast head only)
                c = cf = None
                kl_c = 0.0
                idx = torch.as_tensor(pats, device=device)
                if model.pcodes is not None:  # MAP (prior term 0.5 |c - mu_c|^2) or variational (a draw, KL to N(mu_c, I))
                    c, kl_c = model.pcodes.sample(idx, ex["cp_in"])
                if model.flare is not None:  # the flare codes: variational, KL to N(0, I)
                    cf, kl_f = model.flare.sample_codes(idx)
                    kl_c = kl_c + kl_f
                rec, kl, pred, rate = sde_terms(model, ex, c, cf)
                loss = loss + args.w_sde * (rec + rate + args.beta * kl) + args.w_pred * pred
                if args.w_roll > 0:  # the multi-step rollout loss (12 months, against the later scheduled visits)
                    roll = rollout_loss(model, ex, roll_batch(data, pats, rng, device), c, cf)
                    loss = loss + args.w_roll * roll
                if model.pcodes is not None or model.flare is not None:  # per step, as the KL of the states
                    loss = loss + args.w_sde * args.beta * kl_c / ex["valid"].sum()
            ri = rr = torch.zeros(())
            if args.w_recon > 0:
                m_post = model.last_post[0]
                ri, rr = view_recon((m_post, m_post) if args.recon_latent else model.codes, (model.dec_img, model.dec_rna), batch,
                                    hidden_too=bool(args.recon_latent))
                loss = loss + args.w_recon * (ri + rr)
            opt.zero_grad()
            if code_opt:
                code_opt.zero_grad()
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(main, 10.0)):
                opt.step()
                if code_opt:
                    code_opt.step()
                if args.l21 > 0:
                    group_prox(token_maps(model), args.l21 * opt.param_groups[0]["lr"], [bool(batch["img_p"].any()), bool(batch["rna_p"].any())])
            sched.step()
            step += 1
            if step % 100 == 0:
                say(log_line(model, args, step, time.time() - start, l1, rec, kl, pred, rate, roll, ri, rr))
            if args.train_pause:
                time.sleep(args.train_pause)
            if step % args.val_every == 0:
                v = float(np.mean(np.abs(predict(model, data, list(val), device)[0] - yv)))
                if v < best:
                    best, best_step, best_state = v, step, {k: x.detach().clone() for k, x in model.state_dict().items()}
                say(f"{step} validation MAE (real labels) {v:.4f} (best {best:.4f} at {best_step})")
                if step - best_step >= args.patience * args.val_every:
                    say(f"no improvement for {args.patience} checks: stop")
                    stop = True
                    break
    if args.keep_last:  # e.g. a fine-tune for the long-run dynamics: the validation MAE of the head is not its target
        say(f"model of the last step {step} is kept (best validation MAE {best:.4f} at {best_step})")
    elif best_state is not None:
        model.load_state_dict(best_state)
        say(f"model of step {best_step} (validation MAE {best:.4f}) is kept")
    return step, best_step, best
