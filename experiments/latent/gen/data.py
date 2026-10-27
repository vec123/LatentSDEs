"""
The records of the generator and the batches. All arrays are sorted by patient, with per-patient offsets (off_v visits, off_f onsets,
off_q visit-flare pairs, off_c rate cells, off_k knots); the indices inside are patient-local (visit -> knot, pair -> visit and onset),
so gather(rec, patients) can cut out any set of patients and re-index it without loops.
"""
import os

import numpy as np
import pandas as pd
import torch

from experiments.latent.flare_model import D, DRUGS, NPZ, REACH, static_features

KNOT, CUT = 30.44, 60.0  # days per knot (one month); the cut window after an onset (days)
CUTOFF = pd.Timestamp("2025-12-31")  # the common end of all records (the labels: mean IHS4 of Jan-Jun 2026)
SCHED = ("background", "treatment_monitoring")
CACHE = "runs/gen_records{}.pt"


def _drug_path(tx, first, T):
    bio, ab = {}, {}
    for p, g in tx.groupby("patient_id"):
        s = (g.start_date - first[p]).dt.days.to_numpy(float)
        e = (g.end_date - first[p]).dt.days.fillna(T[p] + 400).to_numpy(float)  # open courses: past the cutoff
        cls = g.treatment_class.to_numpy()
        bio[p] = [(DRUGS.index(n), a, b) for n, a, b, k in zip(g.drug_name, s, e, cls) if k == "Biologic"]
        ab[p] = np.sort(s[cls == "Antibiotic"])
    return bio, ab


def _rna(pats, pid):
    meta = pd.read_csv(D + "rnaseq_metadata.csv", parse_dates=["collection_date"]).sort_values("collection_date")
    emb = pd.read_csv(D + "rnaseq_embeddings.csv").set_index("rnaseq_sample_id")
    E = emb.to_numpy(np.float64)
    U = np.linalg.svd(E - E.mean(0), full_matrices=False)[0][:, :3]
    pcs = pd.DataFrame(U / U.std(0), index=emb.index)
    R, has = np.zeros((len(pats), 3), np.float32), np.zeros(len(pats), np.float32)
    for p, s in meta.drop_duplicates("patient_id").set_index("patient_id").rnaseq_sample_id.items():
        if p in pid:
            R[pid[p]], has[pid[p]] = pcs.loc[s].to_numpy(), 1.0
    return R, has


def load(n_patients=None):
    """the records of all train-folder patients with >= 1 visit (or a random subset; seed 7: hs_build.py drew the held-out patients with
    seed 0 from the same sorted list)"""
    cache = CACHE.format(f"_{n_patients}" if n_patients else "")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    npz = np.load(NPZ, allow_pickle=True)
    held = dict(zip(npz["patient_id"], npz["is_test_patient"] > 0))
    v = pd.read_csv(D + "visits.csv", parse_dates=["visit_date"]).sort_values(["patient_id", "visit_date"])
    first = v.groupby("patient_id").visit_date.min()
    pats = np.array(sorted(first.index))
    if n_patients:
        pats = np.sort(np.random.default_rng(7).choice(pats, n_patients, replace=False))
    v = v[v.patient_id.isin(set(pats))]
    pid = {p: i for i, p in enumerate(pats)}
    T = (CUTOFF - first).dt.days.astype(float)
    tx = pd.read_csv(D + "treatments.csv", parse_dates=["start_date", "end_date"])
    bio, ab = _drug_path(tx[tx.patient_id.isin(set(pats))], first, T)

    def drug_at(p, t):
        u = np.zeros((len(t), len(DRUGS)), np.float32)
        for k, s, e in bio.get(p, []):
            u[(t >= s) & (t <= e), k] = 1.0
        return u

    V, Fo, Q, C, K, P = [], [], [], [], [], []
    for p, g in v.groupby("patient_id", sort=True):
        Ti = T[p]
        t = (g.visit_date - first[p]).dt.days.to_numpy(float)
        ty, y = g.visit_type.to_numpy(), np.log1p(g.ihs4_score.to_numpy(float))
        tau = t[ty == "flare_reactive"]
        sc = np.flatnonzero(np.isfinite(y))
        lag = t[sc][:, None] - tau[None, :]
        jj, kk = np.nonzero((lag >= 0) & (lag <= REACH))
        win = np.zeros(len(sc), bool)
        win[jj[lag[jj, kk] <= CUT]] = True
        V.append((t[sc], y[sc], (t[sc] // KNOT).astype(np.int64), (t[sc] % KNOT) / KNOT, win, np.isin(ty[sc], SCHED), ty[sc] == "flare_reactive"))
        abn = np.array([np.abs(ab[p] - x).min() <= 7 if len(ab.get(p, [])) else False for x in tau], np.float32)
        Fo.append((tau, drug_at(p, tau), abn))
        Q.append((jj.astype(np.int64), kk.astype(np.int64), lag[jj, kk]))
        edges = np.append(np.arange(0, Ti, KNOT), Ti)
        mid = 0.5 * (edges[1:] + edges[:-1])
        C.append((mid, np.diff(edges), drug_at(p, mid)))
        nk = int(Ti // KNOT) + 2
        K.append((drug_at(p, np.arange(nk) * KNOT), np.full(nk, np.median(y[sc]) if len(sc) else 1.5)))
        P.append((Ti, drug_at(p, np.array([Ti]))[0]))
    cat = lambda lst, k, dt=torch.float32: torch.as_tensor(np.concatenate([x[k] for x in lst]), dtype=dt)
    offs = lambda lst: torch.as_tensor(np.r_[0, np.cumsum([len(x[0]) for x in lst])], dtype=torch.long)
    X = static_features(pd.read_csv(D + "demographics.csv")).loc[pats]
    R, has = _rna(pats, pid)
    lab = pd.read_csv(D + "targets.csv").set_index("patient_id").avg_ihs4_next6.reindex(pats).to_numpy(np.float32)
    rec = dict(pats=pats, held=np.array([held[p] for p in pats]), label=lab,
               x_names=list(X.columns) + ["RNAseq PC1", "RNAseq PC2", "RNAseq PC3", "RNAseq sampled"],
               X=torch.as_tensor(np.column_stack([X.to_numpy(np.float32), R, (has - has.mean()) / has.std()])),
               R=torch.as_tensor(R * has[:, None]), T=torch.tensor([x[0] for x in P], dtype=torch.float32),
               uT=torch.as_tensor(np.array([x[1] for x in P])),
               v_t=cat(V, 0), v_y=cat(V, 1), v_k=cat(V, 2, torch.long), v_f=cat(V, 3), v_win=cat(V, 4, torch.bool), v_sched=cat(V, 5, torch.bool),
               v_flare=cat(V, 6, torch.bool), off_v=offs(V),
               f_t=cat(Fo, 0), f_u=cat(Fo, 1), f_ab=cat(Fo, 2), off_f=offs(Fo),
               q_j=cat(Q, 0, torch.long), q_k=cat(Q, 1, torch.long), q_lag=cat(Q, 2), off_q=offs(Q),
               c_t=cat(C, 0), c_dt=cat(C, 1), c_u=cat(C, 2), off_c=offs(C),
               k_u=cat(K, 0), k_init=cat(K, 1), off_k=offs(K))
    os.makedirs("runs", exist_ok=True)
    torch.save(rec, cache)
    return rec


def to(rec, device):
    return {k: (x.to(device) if torch.is_tensor(x) else x) for k, x in rec.items()}


def _rows(off, pats):
    """the rows of the patients in an array with per-patient offsets: (global row index, batch offsets, batch owner of each row)"""
    lens = off[pats + 1] - off[pats]
    boff = torch.zeros(len(pats) + 1, dtype=torch.long, device=off.device)
    boff[1:] = lens.cumsum(0)
    idx = torch.repeat_interleave(off[pats] - boff[:-1], lens) + torch.arange(int(boff[-1]), device=off.device)
    return idx, boff, torch.repeat_interleave(torch.arange(len(pats), device=off.device), lens)


def gather(rec, pats):
    """the batch of the patients `pats` (LongTensor of patient indices): batch-local indices (v_p, f_p, c_p, k_p: the batch row of the
    patient; v_k, q_j, q_k: rows of the batch arrays) and the global rows of the local variational parameters (k_id knots, f_id onsets)"""
    iv, bv, ov = _rows(rec["off_v"], pats)
    i_f, bf, of = _rows(rec["off_f"], pats)
    iq, _, oq = _rows(rec["off_q"], pats)
    ic, _, oc = _rows(rec["off_c"], pats)
    ik, bk, ok = _rows(rec["off_k"], pats)
    return dict(pats=pats, B=len(pats), X=rec["X"][pats], R=rec["R"][pats],
                v_p=ov, v_t=rec["v_t"][iv], v_y=rec["v_y"][iv], v_k=bk[ov] + rec["v_k"][iv], v_f=rec["v_f"][iv], v_win=rec["v_win"][iv],
                f_p=of, f_t=rec["f_t"][i_f], f_u=rec["f_u"][i_f], f_ab=rec["f_ab"][i_f], f_id=i_f,
                q_j=bv[oq] + rec["q_j"][iq], q_k=bf[oq] + rec["q_k"][iq], q_lag=rec["q_lag"][iq],
                c_p=oc, c_t=rec["c_t"][ic], c_dt=rec["c_dt"][ic], c_u=rec["c_u"][ic],
                k_p=ok, k_u=rec["k_u"][ik], k_id=ik, k_first=bk[:-1], k_next=torch.nonzero(ok[1:] == ok[:-1]).ravel())
