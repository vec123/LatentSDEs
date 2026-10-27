"""
Task 2 with the end-to-end model (hs_e2e.py): the patients with an open biologic line at the cutoff.

For each such patient: the current biologic (the active one with the latest start) and its decision step (the day before the
start). Head 2 from the decision step gives the mean IHS4 of 1 Jan - 30 Jun 2026 without treatment (plan "none": Y0) and with each
drug (Y_d), on the scale of the visit-based labels. Head 1 at the final step gives the treated forecast from all data up to the
cutoff (Y1, median and Laplace scale), on the scale of the real label; it reads the RNAseq and the image. So does head 2 at the final
step, but from the decision step head 2 gains little from the RNAseq (nothing trains the 2026 outcome there, and most RNAseq starts
are in 2025). Y0 goes to the scale of the real label
with a factor a, fitted on training patients that never had a biologic: real label ~ a * (head 2, plan none, horizon 0, final step).
super_responder_prob = P(Y1 < 0.25 a Y0), with Y1 Laplace and Y0 held fixed.
The counterfactual is not observed. Checks on the held-out patients of the train folder:
  1. head 2 against observed windows, per plan and horizon (pairs drawn as in training), against persistence (the last score);
  2. the RNAseq cohort: the predicted benefit Y_d / Y0 at the decision step against the realized response (mean score in days
     90..270 after the start / mean in the 180 days before, as in rna_response.py), with and without the RNAseq;
  3. the distribution of super_responder_prob, by drug and line, and the drivers: the change of log(Y1 / Y0) when a feature group or a
     modality is hidden, and the features that differ most between the likely super-responders and the others.

    python -m experiments.latent.hs_task2 runs/hs_e2e_<stamp>
"""
import argparse

import numpy as np
import torch

from experiments.latent.hs_e2e import DECISION, DRUGS, MONTH, PATHS, Data, make_model, plan_at

GROUPS = {
    "score history": ["mean_3m", "n_3m", "std_3m", "mean_6m", "n_6m", "std_6m", "mean_12m", "n_12m", "std_12m", "mean_prev6m", "n_prev6m",
                      "trend_6m", "mean_all", "n_scores_all", "last_score", "days_since_score", "own_score"],
    "visit pattern": ["time_months", "n_visits_so_far", "days_since_prev_visit", "visit_type", "visit_month", "vis6m_background",
                      "vis6m_flare_reactive", "vis6m_follow_up", "vis6m_treatment_monitoring"],
    "treatment history": ["n_tx_started", "days_since_tx_start", "n_tx_ended", "n_tx_active", "Antibiotic_ever", "Antibiotic_active",
                          "Biologic_ever", "Biologic_active", "max_bio_line", "bio_active_adalimumab", "bio_active_bimekizumab",
                          "bio_active_secukinumab", "days_on_bio", "cur_bio_line"],
    "surgeries": ["n_surg", "surg_recur_rate", "days_since_surg", "last_wound_weeks", "n_surg_1y", "surg_CO2 laser ablation", "surg_Deroofing",
                  "surg_Incision & drainage", "surg_Wide local excision"],
    "demographics": ["birth_year", "age_at_diagnosis", "bmi_at_baseline", "diagnostic_delay_years", "hurley_stage_baseline", "sex", "ethnicity",
                     "smoking_status", "family_history_hs", "age_at_visit", "days_since_diag", "days_since_onset"],
}
PROFILE = ["age_at_visit", "sex", "bmi_at_baseline", "smoking_status", "hurley_stage_baseline", "diagnostic_delay_years", "family_history_hs",
           "days_since_diag", "mean_6m", "mean_all", "last_score", "n_scores_all", "Antibiotic_active", "n_surg", "max_bio_line",
           "vis6m_flare_reactive", "days_since_score"]


def load(run, device):
    ck = torch.load(f"{run}/model.pt", map_location=device)
    data = Data(PATHS["train"], stats=ck["stats"])
    model = make_model(data, ck["args"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, data, ck


def open_lines(data, pats):
    """(patient, start day, drug, local row of the decision step, local row of the final step) for each patient with an active
    biologic at the cutoff; the current line is the active one with the latest start"""
    cut, out = data.end_day, []
    for i in pats:
        st, en, dr = data.bio[i]
        act = (st <= cut) & (en >= cut)
        if not act.any():
            continue
        j = np.flatnonzero(act)[np.argmax(st[act])]
        s0, s1 = data.off[i], data.off[i + 1]
        dec = np.flatnonzero((data.kind[s0:s1] == DECISION) & (data.day[s0:s1] == st[j] - 1))
        if len(dec):
            out.append((int(i), int(st[j]), int(dr[j]), int(dec[0]), int(data.final_row(i) - s0)))
    return out


@torch.no_grad()
def counterfactual(model, data, lines, device, chunk=50, **hide):
    """From the decision step: Y0 and Y_d (3 drugs), head 2. At the final step: head 2 with the current drug (y1_h2) and head 1
    (y1, b1: median and scale on the scale of the real label). Label units."""
    ys, ym, cut, res = data.stats["ystd"], data.stats["ymean"], data.end_day, []
    for a in range(0, len(lines), chunk):
        part = lines[a:a + chunk]
        h = model.encode(data.pack([p[0] for p in part], device, **hide))
        n = len(part)
        b = torch.arange(n, device=device)
        kd, kf = torch.as_tensor([p[3] for p in part], device=device), torch.as_tensor([p[4] for p in part], device=device)
        s = np.array([p[1] for p in part], np.float64)
        f = lambda v: torch.as_tensor(np.asarray(v, np.float32), device=device)
        hz, tau, zero = f((cut - (s - 1)) / MONTH), f((cut - s) / MONTH), torch.zeros(n, device=device)
        plan = lambda k: torch.full((n,), k, dtype=torch.long, device=device)
        one = torch.ones(n, device=device) if model.v3 else None  # v3: head 2 on the scale of the real label
        y0 = plan_at(model, h, b, kd, hz, zero, plan(0), real=one)[0]
        yd = [plan_at(model, h, b, kd, hz, tau, plan(d + 1), real=one)[0] for d in range(len(DRUGS))]
        y1_h2 = plan_at(model, h, b, kf, zero, tau, torch.as_tensor([p[2] + 1 for p in part], device=device), real=one)[0]
        y1, b1 = model.forecast(h[b, kf])
        to = lambda t: (t * ys + ym).cpu().numpy()
        res.append(np.column_stack([to(y0)] + [to(v) for v in yd] + [to(y1_h2), to(y1), (b1 * ys).cpu().numpy()]))
    R = np.concatenate(res)
    return {"y0": R[:, 0], "yd": R[:, 1:1 + len(DRUGS)], "y1_h2": R[:, -3], "y1": R[:, -2], "b1": R[:, -1]}


@torch.no_grad()
def scale_factor(model, data, pats, device, chunk=50):
    """a in: real label ~ a * (head 2, plan none, horizon 0, at the final step), least squares on patients without any biologic"""
    if model.v3:  # head 2 is on the scale of the real label already
        return 1.0, 0
    j = data.names.index("Biologic_ever")
    ok = [i for i in pats if data.raw[data.final_row(i), j] == 0 and data.ym[data.final_row(i)]]
    ys, ym, num, den = data.stats["ystd"], data.stats["ymean"], 0.0, 0.0
    for a in range(0, len(ok), chunk):
        part = ok[a:a + chunk]
        h = model.encode(data.pack(part, device))
        n = len(part)
        kf = torch.as_tensor([data.final_row(i) - data.off[i] for i in part], device=device)
        z = torch.zeros(n, device=device)
        y0 = (plan_at(model, h, torch.arange(n, device=device), kf, z, z, torch.zeros(n, dtype=torch.long, device=device))[0] * ys + ym).cpu().numpy()
        yr = np.array([data.y[data.final_row(i)] for i in part])
        num, den = num + float(np.sum(yr * y0)), den + float(np.sum(y0 * y0))
    return num / den, len(ok)


def sr_prob(y1, b1, y0, frac=0.25):
    """P(Y1 < frac * Y0) for Y1 ~ Laplace(y1, b1)"""
    x = frac * np.maximum(y0, 0.0)
    return np.where(x < y1, 0.5 * np.exp((x - y1) / b1), 1 - 0.5 * np.exp(-(x - y1) / b1))


@torch.no_grad()
def check_plan_head(model, data, pats, device, rng, k=20, chunk=50):
    """head 2 and persistence against the observed windows of (step, horizon) pairs with the plan of the patient's record"""
    ys, ym, rows = data.stats["ystd"], data.stats["ymean"], []
    j_last = data.names.index("last_score")
    for a in range(0, len(pats), chunk):
        sel = pats[a:a + chunk]
        ex = [(b,) + e for b, i in enumerate(sel) for e in data.plan_examples(i, rng, k, partial=False)]
        if not ex:
            continue
        E = np.array(ex, dtype=np.float64)
        h = model.encode(data.pack(sel, device))
        f = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=device)
        mu = plan_at(model, h, f(E[:, 0], torch.long), f(E[:, 1], torch.long), f(E[:, 2]), f(E[:, 3]), f(E[:, 4], torch.long))[0]
        g = np.array([data.off[sel[int(b)]] + int(kk) for b, kk in E[:, :2]])
        last = np.where(data.M[g, j_last], data.raw[g, j_last], ym)
        rows.append(np.column_stack([E[:, 4], E[:, 2], E[:, 5], (mu * ys + ym).cpu().numpy(), last, data.kind[g]]))
    R = np.concatenate(rows)
    print(f"\n1. head 2 against observed windows of the held-out patients ({len(R)} (step, horizon) pairs; MAE)")
    print(f"   {'plan':12s} {'horizon (months)':>17s} {'n':>6s} {'head 2':>7s} {'persistence':>12s}")
    for p in range(1 + len(DRUGS)):
        for lo, hi in ((0, 6), (6, 12), (12, 36), (36, 120)):
            s = (R[:, 0] == p) & (R[:, 1] >= lo) & (R[:, 1] < hi)
            if s.sum() >= 30:
                print(f"   {(['none'] + DRUGS)[p]:12s} {f'{lo}-{hi}':>17s} {s.sum():6d} {np.mean(np.abs(R[s, 3] - R[s, 2])):7.3f} {np.mean(np.abs(R[s, 4] - R[s, 2])):12.3f}")
    s = R[:, 5] == DECISION
    print(f"   from decision steps only: n={s.sum()}, head 2 {np.mean(np.abs(R[s, 3] - R[s, 2])):.3f}, persistence {np.mean(np.abs(R[s, 4] - R[s, 2])):.3f}")


def auc(score, label):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


@torch.no_grad()
def check_rna_response(model, data, pats, device, title):
    """the starts with a baseline RNAseq sample: predicted Y_d / Y0 (window 90..270 days after the start) against the realized ratio"""
    starts = []
    for i in pats:
        s0, s1 = data.off[i], data.off[i + 1]
        st, en, dr = data.bio[i]
        sd, sv = data.day[s0:s1][data.has_score[s0:s1]], data.score[s0:s1][data.has_score[s0:s1]]
        for kd in np.flatnonzero((data.kind[s0:s1] == DECISION) & (data.rna_row[s0:s1] >= 0)):
            s = data.day[s0 + kd] + 1
            j = np.flatnonzero(st == s)
            if not len(j):  # the post window may end after the data (its scores up to the end count), as in rna_response.py
                continue
            pre, post = sv[(sd >= s - 180) & (sd < s)], sv[(sd >= s + 90) & (sd <= s + 270)]
            if len(pre) and len(post) and pre.mean() >= 3:
                starts.append((int(i), int(s), int(dr[j[0]]), int(kd), post.mean() / pre.mean()))
    if len(starts) < 10:
        print(f"\n2. {title}: only {len(starts)} starts with a baseline RNAseq sample and both windows; skipped")
        return
    ratio = np.array([s[4] for s in starts])
    good = ratio < 0.5
    out = {}
    for hide in (False, True):
        pred = []
        for a in range(0, len(starts), 50):
            part = starts[a:a + 50]
            h = model.encode(data.pack([p[0] for p in part], device, hide_rna=hide))
            n = len(part)
            bi, kd = torch.arange(n, device=device), torch.as_tensor([p[3] for p in part], device=device)
            hz = torch.full((n,), (90 + 1) / MONTH, device=device)
            y0 = plan_at(model, h, bi, kd, hz, torch.zeros(n, device=device), torch.zeros(n, dtype=torch.long, device=device))[0]
            yd = plan_at(model, h, bi, kd, hz, torch.full((n,), 90 / MONTH, device=device), torch.as_tensor([p[2] + 1 for p in part], device=device))[0]
            ys, ym = data.stats["ystd"], data.stats["ymean"]
            pred.append(((yd * ys + ym) / (y0 * ys + ym).clamp(min=0.5)).cpu().numpy())
        out[hide] = np.concatenate(pred)
    print(f"\n2. {title}: {len(starts)} starts with a baseline RNAseq sample; good responders (realized ratio < 0.5): {good.sum()}")
    for hide, lab in ((False, "with RNAseq"), (True, "RNAseq hidden")):
        r = out[hide]
        print(f"   {lab:14s} AUC of -predicted ratio {auc(-r, good):.3f}   corr(predicted, realized ratio) {np.corrcoef(r, ratio)[0, 1]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, data, ck = load(args.run, device)
    te = list(np.flatnonzero(data.flag))
    tr = list(np.flatnonzero(~data.flag))
    rng = np.random.default_rng(0)
    check_plan_head(model, data, te, device, rng)
    check_rna_response(model, data, te, device, "held-out patients")
    check_rna_response(model, data, tr, device, "training patients (in sample: optimistic)")

    a, n_a = scale_factor(model, data, tr, device)
    print(f"\nscale factor of the untreated forecast: real label ~ {a:.3f} * head 2 (plan none), fitted on {n_a} training patients without a biologic")
    lines = open_lines(data, te)
    cf = counterfactual(model, data, lines, device)
    p = sr_prob(cf["y1"], cf["b1"], a * cf["y0"])
    ratio = cf["y1"] / np.maximum(a * cf["y0"], 0.5)
    p_h2 = sr_prob(cf["y1_h2"], cf["b1"] / a, cf["y0"])
    yreal = np.array([data.y[data.off[l[0]] + l[4]] for l in lines])
    print(f"   MAE of Y1 against the real label: head 1 {np.mean(np.abs(cf['y1'] - yreal)):.3f}, head 2 (x a) {np.mean(np.abs(a * cf['y1_h2'] - yreal)):.3f}; "
          f"probability from head 2 only: mean {p_h2.mean():.3f}, corr with the head-1 probability {np.corrcoef(p, p_h2)[0, 1]:+.3f}")
    drug = np.array([l[2] for l in lines])
    fin = np.array([data.off[l[0]] + l[4] for l in lines])
    line = np.nan_to_num(data.raw[fin, data.names.index("cur_bio_line")], nan=0)
    print(f"\n3. held-out patients with an open biologic line: {len(lines)}")
    print(f"   predicted Y1 / Y0: median {np.median(ratio):.2f}, [10%, 90%] = [{np.percentile(ratio, 10):.2f}, {np.percentile(ratio, 90):.2f}]; "
          f"share below 0.25 {np.mean(ratio < 0.25):.3f}; super_responder_prob: mean {p.mean():.3f}, share > 0.5 {np.mean(p > 0.5):.3f}")
    for d in range(len(DRUGS)):
        s = drug == d
        if s.any():
            print(f"   {DRUGS[d]:12s} n={s.sum():5d}  median ratio {np.median(ratio[s]):.2f}  mean prob {p[s].mean():.3f}")
    for ln in (1, 2, 3):
        s = line == ln
        if s.any():
            print(f"   line {ln:<7d} n={s.sum():5d}  median ratio {np.median(ratio[s]):.2f}  mean prob {p[s].mean():.3f}")
    rna = np.array([data.rna_row[data.off[l[0]] + l[4]] >= 0 for l in lines])
    if rna.sum() >= 5:
        print(f"   with an RNAseq sample: n={rna.sum()}  median ratio {np.median(ratio[rna]):.2f}  mean prob {p[rna].mean():.3f}; "
              f"without: median ratio {np.median(ratio[~rna]):.2f}  mean prob {p[~rna].mean():.3f}")
    best = np.argmin(cf["yd"] / np.maximum(cf["y0"], 0.5)[:, None], 1)
    print(f"   the drug with the lowest predicted Y_d / Y0 is the given drug for {np.mean(best == drug):.2f} of the patients; "
          f"best drug counts {dict(zip(DRUGS, np.bincount(best, minlength=3).tolist()))}")

    print("\n   drivers: mean |change of log(Y1 / a Y0)| and change of the mean probability when a group is hidden")
    base = np.log(np.maximum(cf["y1"], 0.1) / np.maximum(a * cf["y0"], 0.5))
    hides = {g: {"hide_cols": [data.names.index(c) for c in cols if c in data.names]} for g, cols in GROUPS.items()}
    hides.update({"image": {"hide_img": True}, "RNAseq": {"hide_rna": True}})
    for g, kw in hides.items():
        c = counterfactual(model, data, lines, device, **kw)
        lr = np.log(np.maximum(c["y1"], 0.1) / np.maximum(a * c["y0"], 0.5))
        print(f"   {g:18s} {np.mean(np.abs(lr - base)):.3f}   {sr_prob(c['y1'], c['b1'], a * c['y0']).mean() - p.mean():+.3f}")
    print("\n   profile at the decision step: likely super-responders (prob > 0.5) against the others (standardized mean difference)")
    dec = np.array([data.off[l[0]] + l[3] for l in lines])
    hi_ = p > 0.5
    if 5 <= hi_.sum() <= len(p) - 5:
        smd = []
        for c in PROFILE:
            j = data.names.index(c)
            v = np.where(data.M[dec, j], data.raw[dec, j], np.nan)
            a, b = v[hi_], v[~hi_]
            sd = np.nanstd(v)
            if sd > 0:
                smd.append((c, (np.nanmean(a) - np.nanmean(b)) / sd, np.nanmean(a), np.nanmean(b)))
        smd.append(("drug = Adalimumab", (np.mean(drug[hi_] == 0) - np.mean(drug[~hi_] == 0)) / np.std(drug == 0), np.mean(drug[hi_] == 0), np.mean(drug[~hi_] == 0)))
        for c, s_, a, b in sorted(smd, key=lambda r: -abs(r[1]))[:12]:
            print(f"   {c:24s} {s_:+.2f}   ({a:.2f} vs {b:.2f})")
    else:
        print(f"   too few likely super-responders ({hi_.sum()}) for a profile")


if __name__ == "__main__":
    main()
