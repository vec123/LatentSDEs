"""
Evaluation of the SDE part: the rollout forecast (Euler-Maruyama / exact-OU paths on a 7-day grid, treatment of the step held, drawn
flares), the flare report, and evaluate_sde (code fits of held-out patients, rollouts, paired comparisons of the head with the reference
models, attention, embedding dimensions in use, the measurement model).
"""
import time

import numpy as np
import torch

from experiments.latent.fused.batch import STEP_DAYS, extras
from experiments.latent.fused.inference import fit_codes
from experiments.latent.fused.objective import flare_terms
from experiments.latent.hs_e2e import MONTH, QUERY, predict
from experiments.latent.parts.flares import rollout_step
from experiments.latent.parts.stats import boot

# reference runs on the same held-out patients, with and without ethnicity in the input (paired comparisons)
REFS = {True: (("GRU raw (hs_e2e.py --raw 1)", "runs/hs_e2e_20261026-174232"), ("XGBoost tuned (features)", "runs/xgb_e2e_tuned")),
        False: (("GRU (hs_e2e.py, no ethnicity)", "runs/hs_e2e_20261025-142654"), ("XGBoost tuned (no ethnicity)", "runs/xgb_e2e_cv5_fold0"))}


@torch.no_grad()
def rollout_forecast(model, data, pairs, device, n_traj=32, months=6.0, chunk=50, gen=None, codes=None, noise=True):
    """SDE forecast of the mean score over (t, t + months] from the posterior at step k of patient i, for pairs (i, k): paths on a 7-day
    grid with the treatment flags of the step held, the decoded median score averaged over the grid points; the median over the paths
    (visit scale); codes: the patient code of each pair (models with codes; None: the prior mean; flare models: [code | flare code]); noise
    False: the mean path (one path from the posterior mean, drift only, the expected flare burden)"""
    model.eval()
    n_traj = n_traj if noise else 1
    fl = model.flare
    need = {}
    for j, (i, k) in enumerate(pairs):
        need.setdefault(i, []).append((j, k))
    pats = sorted(need)
    f = lambda a: torch.as_tensor(a, device=device)
    if fl is not None:  # the excess of the observed flares at the start step F0 (mean amplitudes), then decay
        cf_all = np.asarray(codes, np.float32)[:, model.d_code:] if codes is not None else np.zeros((len(pairs), 3), np.float32)
        codes = (np.asarray(codes, np.float32)[:, :model.d_code] if model.d_code else None) if codes is not None else None
        F0 = np.zeros(len(pairs), np.float32)
    Mz, Lz = np.zeros((len(pairs), model.dz), np.float32), np.zeros((len(pairs), model.dz), np.float32)
    for a in range(0, len(pats), chunk):
        sel = pats[a:a + chunk]
        model.encode(data.pack(sel, device))
        m, lv = (t.cpu().numpy() for t in model.last_post)
        if fl is not None:
            Fp = flare_terms(model, extras(data, sel, device), f(np.stack([cf_all[need[i][0][0]] for i in sel])), sample=False)["F_post"]
            Fp = Fp.cpu().numpy()
        for bi, i in enumerate(sel):
            for j, k in need[i]:
                Mz[j], Lz[j] = m[bi, k], lv[bi, k]
                if fl is not None:
                    F0[j] = Fp[bi, k]
    rows = np.array([data.off[i] + k for i, k in pairs])
    U, S = data.u[rows], data.s[[i for i, _ in pairs]]
    n_steps, h = int(round(months * MONTH / STEP_DAYS)), STEP_DAYS / MONTH
    lm, ls = data.stats["lm"], data.stats["ls"]
    out = np.zeros(len(pairs))
    for a in range(0, len(pairs), 250):  # 250 pairs x 32 paths per batch (GPU load)
        sl = slice(a, a + 250)
        m_, lv_ = f(Mz[sl]), f(Lz[sl])
        n = m_.shape[0]
        eps = torch.randn((n, n_traj, model.dz), device=device, generator=gen) if noise else torch.zeros((n, 1, model.dz), device=device)
        z = (m_[:, None] + (0.5 * lv_[:, None]).exp() * eps).reshape(n * n_traj, -1)
        u, s = f(U[sl]).repeat_interleave(n_traj, 0), f(S[sl]).repeat_interleave(n_traj, 0)
        c = f(np.asarray(codes[sl], np.float32)).repeat_interleave(n_traj, 0) if codes is not None else None
        acc = torch.zeros(n * n_traj, device=device)
        F = torch.zeros(n * n_traj, device=device)
        if fl is not None:
            sc = fl.effects(f(cf_all[sl]).repeat_interleave(n_traj, 0))
            pA, logD = fl.prior(u, s, sc)
            decay, sA = torch.exp(-STEP_DAYS / logD.exp()), fl.ls_A.exp()
            F = f(F0[sl]).repeat_interleave(n_traj, 0)
        for _ in range(n_steps):
            zm, _, add = model.step(z, torch.zeros_like(z), u, s, h, c)
            z = (zm + (add.sqrt() * torch.randn(z.shape, device=device, generator=gen) if noise else 0.0)).clamp(-20.0, 20.0)
            if fl is not None:  # onsets drawn from the rate of the state (noise False: the expected flare burden instead)
                F = rollout_step(F, decay, fl.onset_prob(z, u, s, sc, STEP_DAYS), pA, sA, draw=noise, gen=gen)
            acc += torch.expm1(model.decode(z)[0] * ls + lm + F).clamp(min=0.0)
        out[sl] = (acc / n_steps).view(n, n_traj).median(1).values.cpu().numpy()
    model.train()
    return out


@torch.no_grad()
def flare_report(model, data):
    """the fitted flare part as factors (the maps are linear: exp(coefficient) per unit of the input; static inputs standardized)"""
    fl = model.flare
    W = lambda lin: lin.weight[0].detach().cpu().numpy()
    B = lambda lin: float(lin.bias.detach().cpu())
    wr, wa, wd, dz = W(fl.rate), W(fl.amp_prior), W(fl.dur), model.dz
    sc = fl.ls_c.exp().cpu().numpy()
    out = [f"\nflare part: rate {np.exp(B(fl.rate)):.2f} per year at z = 0 (no biologic, static 0); |w_z| {np.linalg.norm(wr[:dz]):.3f}; "
           f"amplitude median x{np.exp(np.exp(B(fl.amp_prior))):.2f} on (1 + score), sd of log A {fl.ls_A.exp().item():.2f}; "
           f"half-life {np.log(2) * np.exp(B(fl.dur)):.0f} days (no antibiotic, no biologic); patient sd (frequency, amplitude, "
           f"duration) {sc[0]:.2f} / {sc[1]:.2f} / {sc[2]:.2f} (x{np.exp(2 * 1.2816 * sc[0]):.2f} between the 10th and 90th percentile rate)",
           f"   {'':26s} {'rate':>7s} {'amplitude':>10s} {'duration':>9s}   (factors)"]
    for k, g in enumerate(["antibiotic", "adalimumab", "bimekizumab", "secukinumab"]):
        out.append(f"   {g:26s} {'-' if k == 0 else f'{np.exp(wr[dz + k - 1]):.2f}':>7s} {np.exp(wa[k]):10.2f} {np.exp(wd[k]):9.2f}")
    for j, nm in enumerate(data.s_names):
        if not nm.endswith("missing"):
            out.append(f"   {nm[:26]:26s} {np.exp(wr[dz + 3 + j]):7.2f} {np.exp(wa[4 + j]):10.2f} {np.exp(wd[4 + j]):9.2f}")
    return out


def evaluate_sde(model, data, device, run_dir, max_patients=None, rollout=True, fit_steps=100, quarter=True):
    """the SDE rollout forecast against the head (rollout False: skipped, e.g. without SDE training), and the head against the GRU and
    XGBoost (paired, held-out patients); models with patient codes: the codes of the held-out patients are fitted on their data up to
    the cutoff (fit_codes), the training patients use their trained codes; quarter False: no rollout at the quarter ends (5,000 rows)"""
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    te = list(data.test_idx[:max_patients]) if max_patients else list(data.test_idx)
    yt = np.array([data.y[data.final_row(i)] for i in te])
    head = np.load(f"{run_dir}/final_preds.npz")["full_mu"]
    fr = np.array([data.final_row(i) for i in te])
    roll, mpath, off, q, rq, hq = np.zeros(0), np.zeros(0), np.nan, np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    code_te = code_tr = None
    te_all, yt_all, head_all, fr_all = te, yt, head, fr
    if rollout:
        gen = torch.Generator(device=device).manual_seed(0)
        fin = lambda pats: [(i, data.final_row(i) - data.off[i]) for i in pats]
        trp = [int(i) for i in np.random.default_rng(0).choice(data.train_idx, min(2000, len(data.train_idx)), replace=False)]
        if model.d_code or model.flare is not None:  # the codes are fitted per patient: the rollout test on 1,000 held-out patients
            sub = np.sort(np.random.default_rng(0).choice(len(te), min(1000, len(te)), replace=False))
            te, yt, head, fr = [te[j] for j in sub], yt[sub], head[sub], fr[sub]
            t0 = time.time()
            code_te = fit_codes(model, data, te, device, steps=fit_steps)
            code_tr = np.concatenate([w.weight.detach()[trp].cpu().numpy() for w in ((model.pcode,) if model.d_code else ()) +
                                      ((model.pflare,) if model.flare is not None else ())], 1)
            say(f"\npatient codes: fitted for {len(te)} held-out patients ({fit_steps} Adam steps, {time.time() - t0:.0f}s); |c| held-out "
                f"{np.linalg.norm(code_te, axis=1).mean():.2f}, training {np.linalg.norm(code_tr, axis=1).mean():.2f}")
        roll = rollout_forecast(model, data, fin(te), device, gen=gen, codes=code_te)
        mpath = rollout_forecast(model, data, fin(te), device, codes=code_te, noise=False)
        ytr = np.array([data.y[data.final_row(i)] for i in trp])
        off = float(np.median(ytr - rollout_forecast(model, data, fin(trp), device, gen=gen, codes=code_tr)))
        j_last = data.names.index("last_score")
        last = np.where(data.M[fr, j_last], data.raw[fr, j_last], np.nan)
        say(f"\nSDE rollout forecast (32 paths, 7-day grid, treatment of the step held), held-out final steps (real labels), n={len(te)}")
        say(f"   head on the posterior                      MAE {np.abs(head - yt).mean():.3f}")
        say(f"   SDE rollout, no offset {'(g(z): no visit effects)' if model.meas else '(visit scale)'}  MAE {np.abs(roll - yt).mean():.3f}, "
            f"mean error {np.mean(roll - yt):+.3f}")
        say(f"   SDE rollout + offset {off:+.2f} (training patients) MAE {np.abs(roll + off - yt).mean():.3f}")
        say(f"   SDE mean path (no noise, no offset)        MAE {np.abs(mpath - yt).mean():.3f}, mean error {np.mean(mpath - yt):+.3f}, "
            f"corr {np.corrcoef(mpath, yt)[0, 1]:.2f}; noisy - mean path {np.abs(roll - mpath).mean():.2f} per patient")
        ok = np.isfinite(last)
        frt = np.array([data.final_row(i) for i in trp])
        o2 = float(np.nanmedian(ytr - np.where(data.M[frt, j_last], data.raw[frt, j_last], np.nan)))
        say(f"   last score + offset {o2:+.2f} (patients with a score, n={ok.sum()}) MAE {np.abs(last[ok] + o2 - yt[ok]).mean():.3f} "
            f"(head on them {np.abs(head[ok] - yt[ok]).mean():.3f}, rollout + offset {np.abs(roll[ok] + off - yt[ok]).mean():.3f})")
        if quarter:  # quarter ends of the held-out patients: visit-scale derived labels, the same definition as the rollout (no offset)
            q = np.flatnonzero(np.isin(data.prow, te) & (data.kind == QUERY) & data.ym)
            q = np.sort(np.random.default_rng(1).choice(q, min(5000, len(q)), replace=False))
            pos = {p: k for k, p in enumerate(te)}
            cq = code_te[[pos[int(data.prow[r])] for r in q]] if code_te is not None else None
            rq = rollout_forecast(model, data, [(int(data.prow[r]), int(r - data.off[data.prow[r]])) for r in q], device, gen=gen, codes=cq)
            hq = predict(model, data, te, device, rows="all")[q]
            lq = np.where(data.M[q, j_last], data.raw[q, j_last], np.nan)
            okq = np.isfinite(lq)
            say(f"   quarter-end derived labels (visit scale), n={len(q)}: head MAE {np.abs(hq - data.y[q]).mean():.3f}, SDE rollout MAE "
                f"{np.abs(rq - data.y[q]).mean():.3f}; last score (n={okq.sum()}) {np.abs(lq[okq] - data.y[q][okq]).mean():.3f}"
                + (" (codes fitted up to the cutoff, later than these steps: optimistic)" if cq is not None else ""))
    te_roll = te
    te, yt, head, fr = te_all, yt_all, head_all, fr_all
    nm = data.names.index  # paired: the head against the GRU and XGBoost of the same held-out patients
    groups = {"all": np.ones(len(te), bool), "without visits": data.raw[fr, nm("n_visits_so_far")] == 0,
              "open biologic line": data.raw[fr, nm("Biologic_active")] > 0, "with image": data.img_row[fr] >= 0,
              "with image + RNAseq": (data.img_row[fr] >= 0) & (data.rna_row[fr] >= 0)}
    say("\nhead against the models of the same held-out patients: MAE, and the difference with a 95% interval over patients")
    pos = {p: k for k, p in enumerate(data.pid[te])}
    for name, path in REFS["ethnicity" not in data.stats.get("drop", [])]:
        o = np.load(f"{path}/final_preds.npz")
        other = np.full(len(te), np.nan)
        for p, v in zip(o["patient_id"], o["full_mu"]):
            if p in pos:
                other[pos[p]] = v
        for g, s in groups.items():
            s = s & np.isfinite(other)
            if s.sum() < 10:
                continue
            e = np.abs(head - yt)[s] - np.abs(other - yt)[s]
            lo, hi = boot(e)
            say(f"   {name:30s} {g:20s} n={s.sum():5d}  {np.abs(head - yt)[s].mean():.3f} vs {np.abs(other - yt)[s].mean():.3f}: "
                f"{e.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    say("\nattention of the clinical token at the final step (mean over heads and patients): null / image / RNAseq")
    for g in ("with image", "with image + RNAseq"):
        sel = [te[j] for j in np.flatnonzero(groups[g])]
        W = []
        with torch.no_grad():
            for a in range(0, len(sel), 50):
                part = [int(i) for i in sel[a:a + 50]]
                model.encode(data.pack(part, device))
                W += [model.last_att[bi, data.final_row(i) - data.off[i]].cpu().numpy() for bi, i in enumerate(part)]
        if W:
            say(f"   {g:20s} n={len(W):4d}  " + " / ".join(f"{x:.3f}" for x in np.mean(W, 0)))
    ni, nr = (W.norm(dim=0).detach().cpu().numpy() for W in (model.t_img.weight, model.t_rna.weight))
    say(f"\nembedding dimensions in use (token-map column norm > 0): image {(ni > 0).sum()} of {len(ni)}, RNAseq {(nr > 0).sum()} of {len(nr)}")
    if model.meas:
        bias = (model.vt_bias - model.vt_bias[0]).detach().cpu().numpy()
        scl = torch.exp(model.vt_logs - model.vt_logs[0]).detach().cpu().numpy()
        say("measurement model, visit type effects relative to background visits (factor on 1 + score; scale factor): " +
            ", ".join(f"{t} x{np.exp(bias[k] * data.stats['ls']):.2f} / x{scl[k]:.2f}"
                      for k, t in enumerate(["background", "flare_reactive", "follow_up", "treatment_monitoring"])))
    if model.flare is not None:
        for line in flare_report(model, data):
            say(line)
    with open(f"{run_dir}/sde_eval.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    np.savez(f"{run_dir}/rollout_preds.npz", patient_id=data.pid[te], y=yt, head=head, rollout=roll, rollout_patient_id=data.pid[te_roll],
             mean_path=mpath, offset=off, q_rows=q, q_rollout=rq, q_head=hq, col_norm_img=ni, col_norm_rna=nr,
             **({"codes_heldout": code_te} if code_te is not None else {}))
