"""
Per-patient step sequences of one data folder (train, test_1, test_2) for the end-to-end model (hs_e2e.py).

Steps of a patient, in time order (on one date the visits come first, then a query step, then a decision step):
  visit     one per row of visits.csv
  query     each calendar quarter end (31 Mar, 30 Jun, 30 Sep, 31 Dec) from the first visit on, before the cutoff: no observation
  final     the cutoff 2025-12-31, the time of the challenge prediction (the target is the mean IHS4 of 1 Jan - 30 Jun 2026)
  decision  the day before each biologic start: what is known before the first dose, plus the samples taken on the start day (all
            RNAseq samples are taken on the day of a biologic start: the baseline sample). No label; the treatment head of the
            model starts its counterfactual from this step.
A patient without any visit has the final step (and decision steps) only.

Features of a step are known on its date (causal). They are the visit features of Data_Analysis/xgb_visit_forecast.py (the same
definitions and the same code, here evaluated at every step), without its 16 principal components of the embeddings (the model
reads the full embeddings). New: the step kind, the active biologics of each drug, the line of the current biologic and the days on
it. Categories (visit type, sex, ...) use the codes of the train folder, so that all folders agree.
Embeddings: image (256) and RNAseq (128) per sample, full width; each step has the row of the latest sample on or before its date
(-1: none).
Labels: at a visit or query step, the mean score in the 6 months after the step, when the window ends inside the data and holds a
score (the derived label of xgb_visit_forecast.py); at the final step of a train patient, avg_ihs4_next6 of targets.csv (real).
Train split: is_test_patient of the old sequences (hs_sequences_all_light.npz), so that the held-out patients stay the same;
patients that are not in it (no visits) are drawn at the same 20% rate.
Check (train): the features and labels of the visit steps against the old sequences.

    python -I experiments/latent/hs_build.py          (default python: pandas; the output is readable by numpy 1)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("test_data/hs_public_dataset")
OLD = ROOT / "hs_sequences_all_light.npz"
MONTH = 30.44
W3, W6, W12 = 3 * MONTH, 6 * MONTH, 12 * MONTH
BIG = 100_000  # key = patient index * BIG + day: one sorted axis for all patients
EPOCH = pd.Timestamp("2000-01-01")
CUTOFF = pd.Timestamp("2025-12-31")
QUARTERS = [q for q in (pd.Timestamp(y, m, d) for y in range(2016, 2026) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))) if q < CUTOFF]
VISIT, QUERY, FINAL, DECISION = 0, 1, 2, 3
DRUGS = ["Adalimumab", "Bimekizumab", "Secukinumab"]
PROCEDURES = ["CO2 laser ablation", "Deroofing", "Incision & drainage", "Wide local excision"]
CATEGORICAL = ["sex", "ethnicity", "smoking_status", "family_history_hs"]
STATIC = {"birth_year", "age_at_diagnosis", "bmi_at_baseline", "diagnostic_delay_years", "hurley_stage_baseline",
          "sex", "ethnicity", "smoking_status", "family_history_hs"}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:4.0f}s] {msg}", flush=True)


def categories(folder="train"):
    """The category lists of the train folder (sorted, as pandas' astype('category') in xgb_visit_forecast.py)."""
    D = ROOT / "public" / folder
    demo = pd.read_csv(D / "demographics.csv")
    vis = pd.read_csv(D / "visits.csv", usecols=["visit_type"])
    cats = {c: sorted(demo[c].dropna().unique()) for c in CATEGORICAL}
    cats["visit_type"] = sorted(vis.visit_type.dropna().unique())
    return cats


def build(folder, cats):
    D = ROOT / "public" / folder
    vis = pd.read_csv(D / "visits.csv", parse_dates=["visit_date"])
    tx = pd.read_csv(D / "treatments.csv", parse_dates=["start_date", "end_date"])
    sg = pd.read_csv(D / "surgeries.csv", parse_dates=["procedure_date"])
    demo = pd.read_csv(D / "demographics.csv", parse_dates=["symptom_onset_date", "diagnosis_date"]).set_index("patient_id")
    targets = pd.read_csv(D / "targets.csv").set_index("patient_id")["avg_ihs4_next6"] if (D / "targets.csv").exists() else None
    patients = set(pd.read_csv(D / "patient_assignment.csv").patient_id) | set(demo.index) | set(vis.patient_id)
    if targets is not None:
        patients |= set(targets.index)
    patients = sorted(patients)
    assert set(vis.visit_type.dropna()) <= set(cats["visit_type"]), "a visit type that the train folder does not have"
    assert set(tx.treatment_class) <= {"Antibiotic", "Biologic"} and set(tx[tx.treatment_class == "Biologic"].drug_name) <= set(DRUGS)
    assert set(sg.procedure_type) <= set(PROCEDURES)

    # ---------------------------------------------------------------- the steps
    first = vis.groupby("patient_id").visit_date.min()
    qrows = [(p, q) for p, f in first.items() for q in QUARTERS if q >= f]
    starts = tx[(tx.treatment_class == "Biologic") & tx.start_date.notna() & (tx.start_date <= CUTOFF)]
    steps = pd.concat([vis[["patient_id", "visit_id", "visit_date", "visit_type", "ihs4_score"]].assign(kind=VISIT),
                       pd.DataFrame(qrows, columns=["patient_id", "visit_date"]).assign(kind=QUERY),
                       pd.DataFrame({"patient_id": patients, "visit_date": CUTOFF}).assign(kind=FINAL),
                       pd.DataFrame({"patient_id": starts.patient_id.to_numpy(), "visit_date": starts.start_date.to_numpy() - pd.Timedelta(days=1),
                                     "visit_id": starts.treatment_id.to_numpy()}).assign(kind=DECISION)], ignore_index=True)
    steps["visit_id"] = steps.visit_id.fillna("")
    v = steps.sort_values(["patient_id", "visit_date", "kind", "visit_id"]).reset_index(drop=True)
    pid_codes = {p: i for i, p in enumerate(patients)}
    v["pid"] = v.patient_id.map(pid_codes)
    v["day"] = (v.visit_date - EPOCH).dt.days
    v["key"] = v.pid * BIG + v.day
    pstart = (v.pid * BIG).to_numpy()
    q = v.key.to_numpy()
    n = len(v)
    kind = v.kind.to_numpy()
    is_visit = kind == VISIT
    log(f"{folder}: {len(patients)} patients, {n} steps ({is_visit.sum()} visits, {(kind == QUERY).sum()} quarter ends, "
        f"{(kind == FINAL).sum()} final, {(kind == DECISION).sum()} decisions)")

    def to_key(df, date_col):
        pid = df.patient_id.map(pid_codes)
        ok = pid.notna() & df[date_col].notna()
        k = pd.Series(np.nan, index=df.index)
        k[ok] = pid[ok] * BIG + (df.loc[ok, date_col] - EPOCH).dt.days
        return k

    def sorted_events(df, date_col, **values):
        k = to_key(df, date_col)
        d = df.assign(_k=k).dropna(subset=["_k"]).sort_values("_k")
        cs = {name: np.concatenate([[0.0], np.cumsum(d[col].to_numpy(float))]) for name, col in values.items()}
        return d["_k"].to_numpy(), cs, d

    F = {}
    # ---------------------------------------------------------------- severity history (scores of visits only)
    sc = v[is_visit & v.ihs4_score.notna().to_numpy()]
    S_key, S_val = sc.key.to_numpy(), sc.ihs4_score.to_numpy(float)
    S_cs = np.concatenate([[0.0], np.cumsum(S_val)])
    S_cs2 = np.concatenate([[0.0], np.cumsum(S_val ** 2)])

    def score_window(hi_key, lo_key):
        """mean, count, std of the scores with lo_key < key <= hi_key"""
        hi = np.searchsorted(S_key, hi_key, side="right")
        lo = np.searchsorted(S_key, lo_key, side="right")
        cnt = hi - lo
        s, s2 = S_cs[hi] - S_cs[lo], S_cs2[hi] - S_cs2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, s / cnt, np.nan)
            var = np.where(cnt > 1, (s2 - s * s / cnt) / (cnt - 1), np.nan)
        return mean, cnt, np.sqrt(np.clip(var, 0, None))

    for name, w in [("3m", W3), ("6m", W6), ("12m", W12)]:
        F[f"mean_{name}"], F[f"n_{name}"], F[f"std_{name}"] = score_window(q, q - w)
    F["mean_prev6m"], F["n_prev6m"], _ = score_window(q - W6, q - W12)
    F["trend_6m"] = F["mean_6m"] - F["mean_prev6m"]
    F["mean_all"], F["n_scores_all"], _ = score_window(q, pstart - 1)
    idx = np.searchsorted(S_key, q, side="right") - 1
    ci = np.clip(idx, 0, None)
    has_last = (idx >= 0) & (S_key[ci] >= pstart)
    F["last_score"] = np.where(has_last, S_val[ci], np.nan)
    F["days_since_score"] = np.where(has_last, q - S_key[ci], np.nan)
    F["own_score"] = v.ihs4_score.to_numpy(float)  # NaN at a visit without a score and at every query step

    # ---------------------------------------------------------------- visit context
    F["time_months"] = ((v.visit_date - v.patient_id.map(first)).dt.days / MONTH).to_numpy(float)
    F["n_visits_so_far"] = pd.Series(is_visit.astype(int)).groupby(v.pid.to_numpy()).cumsum().to_numpy(float)
    prev = v.day.where(is_visit).groupby(v.pid).shift(1)
    prev = prev.groupby(v.pid).ffill()  # the day of the last visit before the step
    F["days_since_prev_visit"] = (v.day - prev).to_numpy(float)
    vt = pd.Categorical(v.visit_type, categories=cats["visit_type"]).codes.astype(float)
    F["visit_type"] = np.where(vt < 0, np.nan, vt)
    F["visit_month"] = v.visit_date.dt.month.to_numpy(float)
    for t_ in cats["visit_type"]:
        k = np.sort(v.key[is_visit & (v.visit_type == t_).to_numpy()].to_numpy())
        F[f"vis6m_{t_}"] = (np.searchsorted(k, q, side="right") - np.searchsorted(k, q - W6, side="right")).astype(float)
    F["is_query"] = (kind == QUERY).astype(float)
    F["is_final"] = (kind == FINAL).astype(float)
    F["is_decision"] = (kind == DECISION).astype(float)

    # ---------------------------------------------------------------- treatments
    Tk, _, _ = sorted_events(tx.assign(_one=1.0), "start_date", one="_one")
    lo_p = np.searchsorted(Tk, pstart, side="left")
    hi_s = np.searchsorted(Tk, q, side="right")
    F["n_tx_started"] = (hi_s - lo_p).astype(float)
    last_i = hi_s - 1
    F["days_since_tx_start"] = np.where(last_i >= lo_p, q - Tk[np.clip(last_i, 0, None)], np.nan)
    Ek = np.sort(to_key(tx, "end_date").dropna().to_numpy())
    F["n_tx_ended"] = (np.searchsorted(Ek, q, side="left") - np.searchsorted(Ek, pstart, side="left")).astype(float)
    F["n_tx_active"] = F["n_tx_started"] - F["n_tx_ended"]

    def started_ended(t):
        sk = np.sort(to_key(t, "start_date").dropna().to_numpy())
        ek = np.sort(to_key(t, "end_date").dropna().to_numpy())
        started = np.searchsorted(sk, q, side="right") - np.searchsorted(sk, pstart, side="left")
        ended = np.searchsorted(ek, q, side="left") - np.searchsorted(ek, pstart, side="left")
        return started.astype(float), ended.astype(float)

    for cls in ["Antibiotic", "Biologic"]:
        s_, e_ = started_ended(tx[tx.treatment_class == cls])
        F[f"{cls}_ever"], F[f"{cls}_active"] = s_, s_ - e_
    bio = tx[tx.treatment_class == "Biologic"]
    lk, _, ld = sorted_events(bio.dropna(subset=["treatment_line"]), "start_date", line="treatment_line")
    ld = ld.assign(_cm=ld.groupby(ld.patient_id.map(pid_codes))["treatment_line"].cummax())
    hi_l, lo_l = np.searchsorted(lk, q, side="right"), np.searchsorted(lk, pstart, side="left")
    F["max_bio_line"] = np.where(hi_l > lo_l, ld["_cm"].to_numpy()[np.clip(hi_l - 1, 0, None)], np.nan)
    for dname in DRUGS:  # new: the active biologics of each drug
        s_, e_ = started_ended(bio[bio.drug_name == dname])
        F[f"bio_active_{dname.lower()}"] = s_ - e_
    # new: the current biologic = the latest biologic start on or before the step, if it is still active
    bk, ekb = to_key(bio, "start_date"), to_key(bio, "end_date")
    ok = bk.notna().to_numpy()
    o = np.argsort(bk.to_numpy()[ok], kind="stable")
    bks, eks = bk.to_numpy()[ok][o], ekb.to_numpy()[ok][o]
    lines = bio.treatment_line.to_numpy(float)[ok][o]
    j = np.searchsorted(bks, q, side="right") - 1
    jj = np.clip(j, 0, None)
    act = (j >= 0) & (bks[jj] >= pstart) & (np.isnan(eks[jj]) | (eks[jj] >= q))
    F["days_on_bio"] = np.where(act, q - bks[jj], np.nan)
    F["cur_bio_line"] = np.where(act, lines[jj], np.nan)

    # ---------------------------------------------------------------- surgeries
    Gk, Gcs, Gd = sorted_events(sg.assign(_one=1.0), "procedure_date", one="_one", rec="recurrence_within_6mo")
    hi_g, lo_g = np.searchsorted(Gk, q, side="right"), np.searchsorted(Gk, pstart, side="left")
    F["n_surg"] = (hi_g - lo_g).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["surg_recur_rate"] = np.where(F["n_surg"] > 0, (Gcs["rec"][hi_g] - Gcs["rec"][lo_g]) / np.maximum(F["n_surg"], 1), np.nan)
    okg, last_g = hi_g > lo_g, np.clip(hi_g - 1, 0, None)
    F["days_since_surg"] = np.where(okg, q - Gk[last_g], np.nan) if len(Gk) else np.full(n, np.nan)
    F["last_wound_weeks"] = np.where(okg, Gd["wound_healing_weeks"].to_numpy(float)[last_g], np.nan) if len(Gk) else np.full(n, np.nan)
    F["n_surg_1y"] = np.minimum(hi_g - np.searchsorted(Gk, q - W12, side="right").clip(min=0), F["n_surg"]).astype(float)
    for pr in PROCEDURES:
        k = np.sort(to_key(sg[sg.procedure_type == pr], "procedure_date").dropna().to_numpy())
        F[f"surg_{pr}"] = (np.searchsorted(k, q, side="right") - np.searchsorted(k, pstart, side="left")).astype(float)

    # ---------------------------------------------------------------- static demographics
    d = demo.reindex(v.patient_id).reset_index(drop=True)
    for c in CATEGORICAL:
        code = pd.Categorical(d[c], categories=cats[c]).codes.astype(float)
        d[c] = np.where(code < 0, np.nan, code)
    for c in ["birth_year", "age_at_diagnosis", "bmi_at_baseline", "diagnostic_delay_years", "hurley_stage_baseline"] + CATEGORICAL:
        F[c] = d[c].to_numpy(float)
    F["age_at_visit"] = (v.visit_date.dt.year.to_numpy() - d.birth_year.to_numpy()).astype(float)
    F["days_since_diag"] = (v.visit_date - d.diagnosis_date).dt.days.to_numpy(float)
    F["days_since_onset"] = (v.visit_date - d.symptom_onset_date).dt.days.to_numpy(float)

    # ---------------------------------------------------------------- image and RNAseq samples (full width, from their date on)
    def samples(meta_file, emb_file, id_col, date_col, prefix):
        meta = pd.read_csv(D / meta_file, parse_dates=[date_col])
        emb = pd.read_csv(D / emb_file)
        dims = [c for c in emb.columns if c != id_col]
        m = meta[[id_col, "patient_id", date_col]].merge(emb, on=id_col).sort_values(date_col).reset_index(drop=True)
        # a decision step (the day before a start) also sees the samples of the start day
        seen = v.visit_date + pd.to_timedelta(np.where(kind == DECISION, 1, 0), unit="D")
        left = pd.DataFrame({"patient_id": v.patient_id, "seen": seen, "_row": np.arange(n)}).sort_values("seen")
        out = pd.merge_asof(left, m[["patient_id", date_col]].assign(_s=np.arange(len(m))), left_on="seen", right_on=date_col,
                            by="patient_id", direction="backward").sort_values("_row")
        F[f"{prefix}_days_since"] = (out.seen - out[date_col]).dt.days.to_numpy(float)
        return m[dims].to_numpy(np.float32), out._s.fillna(-1).to_numpy(np.int32), m

    img_emb, img_row, img_m = samples("images_metadata.csv", "images_embeddings.csv", "image_id", "capture_date", "img")
    rna_emb, rna_row, rna_m = samples("rnaseq_metadata.csv", "rnaseq_embeddings.csv", "rnaseq_sample_id", "collection_date", "rna")
    log(f"{len(F)} features; {len(img_m)} image samples ({img_m.patient_id.nunique()} patients), "
        f"{len(rna_m)} RNAseq samples ({rna_m.patient_id.nunique()} patients)")

    # ---------------------------------------------------------------- labels
    fwd, fwd_n, _ = score_window(q + W6, q)  # the scores in (t, t + 6 months]
    end_day = (vis.visit_date.max() - EPOCH).days
    y = np.where(((kind == VISIT) | (kind == QUERY)) & (v.day.to_numpy() + W6 <= end_day) & (fwd_n > 0), fwd, np.nan)
    if targets is not None:
        fin = kind == FINAL
        y[fin] = targets.reindex(v.patient_id[fin]).to_numpy(float)

    X = pd.DataFrame(F)
    names = list(X.columns)
    Xv = X.to_numpy(np.float64)
    M = ~np.isnan(Xv)
    pid = v.pid.to_numpy()
    offsets = np.r_[0, np.cumsum(np.bincount(pid, minlength=len(patients)))].astype(np.int64)
    bio_ok = bio.start_date.notna() & bio.patient_id.isin(pid_codes)
    b = bio[bio_ok]
    out = dict(X=np.where(M, Xv, 0).astype(np.float32), M=M.astype(np.uint8), feature_names=np.array(names),
               static_feature=np.array([c in STATIC for c in names], np.uint8), kind=kind.astype(np.uint8), day=v.day.to_numpy(np.int32),
               y=np.where(np.isnan(y), -1, y).astype(np.float32), y_mask=(~np.isnan(y)).astype(np.uint8), offsets=offsets,
               patient_id=np.array(patients), visit_id=v.visit_id.to_numpy().astype(str),
               img_emb=img_emb, img_row=img_row, rna_emb=rna_emb, rna_row=rna_row,
               bio_patient=b.patient_id.map(pid_codes).to_numpy(np.int32), bio_drug=b.drug_name.map({x: i for i, x in enumerate(DRUGS)}).to_numpy(np.int8),
               bio_start=(b.start_date - EPOCH).dt.days.to_numpy(np.int32),
               bio_end=np.where(b.end_date.notna(), (b.end_date - EPOCH).dt.days, 10 ** 6).astype(np.int32),
               bio_line=b.treatment_line.fillna(0).to_numpy(np.int8))
    return out, v


def check_against_old(out, v):
    """The visit steps of the train folder against the old sequences (same order: patient, date, visit id)."""
    d = np.load(OLD)
    old_names = list(d["feature_names"])
    vis_rows = np.flatnonzero(out["kind"] == VISIT)
    assert len(vis_rows) == len(d["X"]), (len(vis_rows), len(d["X"]))
    pid_old = np.repeat(d["patient_id"], np.diff(d["offsets"]))
    assert (out["patient_id"][np.repeat(np.arange(len(out["patient_id"])), np.diff(out["offsets"]))][vis_rows] == pid_old).all()
    worst = []
    for c in old_names:
        if c not in out["feature_names"]:
            continue
        j = list(out["feature_names"]).index(c)
        a, am = out["X"][vis_rows, j], out["M"][vis_rows, j] > 0
        b, bm = d["X"][:, old_names.index(c)], d["M"][:, old_names.index(c)] > 0
        both = am & bm
        worst.append((c, float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0, int((am != bm).sum())))
    bad = [w for w in worst if w[1] > 1e-3 or w[2] > 0]
    log(f"check: {len(worst)} common features; differing: {bad if bad else 'none'}")
    old_y, old_m, old_r = d["y"], d["y_mask"] > 0, d["y_is_real"] > 0
    new_y, new_m = out["y"][vis_rows], out["y_mask"][vis_rows] > 0
    der = old_m & ~old_r
    log(f"check: old derived labels {der.sum()}, labelled in the new steps too {(der & new_m).sum()}, "
        f"max |difference| {np.abs(old_y[der & new_m] - new_y[der & new_m]).max():.2e}; new visit labels without an old label {(new_m & ~old_m).sum()}")


def main():
    cats = categories()
    old = np.load(OLD)
    old_split = dict(zip(old["patient_id"], old["is_test_patient"]))
    for folder in ("train", "test_1", "test_2"):
        out, v = build(folder, cats)
        if folder == "train":
            check_against_old(out, v)
            rng = np.random.default_rng(1)
            out["is_test_patient"] = np.array([old_split[p] if p in old_split else int(rng.random() < 0.2) for p in out["patient_id"]], np.uint8)
            fin = out["kind"] == FINAL
            log(f"split: {out['is_test_patient'].sum()} held-out patients of {len(out['patient_id'])}; real labels {int(out['y_mask'][fin].sum())}")
        else:
            out["is_test_patient"] = np.ones(len(out["patient_id"]), np.uint8)
            a = pd.read_csv(ROOT / "public" / folder / "answers_teamName_exercise1.csv")
            log(f"{folder}: answer file lists {len(a)} patients; all have a sequence: {set(a.patient_id) <= set(out['patient_id'])}")
        for k_, name in ((VISIT, "visit"), (QUERY, "quarter end"), (FINAL, "final"), (DECISION, "decision")):
            s = out["kind"] == k_
            log(f"  {name:11s} steps {s.sum():8d}, labelled {int(out['y_mask'][s].sum()):7d}, mean label "
                f"{out['y'][s & (out['y_mask'] > 0)].mean() if (s & (out['y_mask'] > 0)).any() else float('nan'):.2f}")
        np.savez_compressed(ROOT / f"hs_e2e_{folder}.npz", **out)
        log(f"saved {ROOT / f'hs_e2e_{folder}.npz'}")


if __name__ == "__main__":
    main()
