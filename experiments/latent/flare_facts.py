"""
Flares in the raw visit data (no model; CPU, seconds): the ground truth for a flare model. A flare_reactive visit is a flare that led to a
visit (an observed flare onset).
1. frequency: flare visits per patient-year; dispersion of the counts over patients (Poisson: var / mean 1; more: a patient frailty)
2. clustering: gaps between the flare visits of a patient (exponential gaps: CV 1; > 1: flares come in clusters)
3. amplitude: the score at a flare visit - the last scored background visit in the 180 days before it
4. decay: the excess of later visits (no new flare visit between), (score - baseline) / (flare score - baseline), by days after the flare
5. antibiotics: the start of the nearest antibiotic course relative to a flare visit
6. biologics: flare visits and background visits per year on and off a biologic (patients with >= 180 days of each)
7. visit noise: practitioners per patient; near-replicate pairs (successive scored scheduled visits <= 21 days apart, no flare visit in
   the 90 days before the first or between them): mean |change of log1p(score)| = 1.5 x the Laplace scale (if the state does not move),
   for pairs with the same and with another practitioner (another practitioner adds 2 x the variance of a rater bias)

    python -m experiments.latent.flare_facts
"""
import numpy as np
import pandas as pd

D = "test_data/hs_public_dataset/public/train/"
T0 = pd.Timestamp("2000-01-01")
YEAR = 365.25


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    v = pd.read_csv(D + "visits.csv", parse_dates=["visit_date"]).sort_values(["patient_id", "visit_date"])
    tx = pd.read_csv(D + "treatments.csv", parse_dates=["start_date", "end_date"])
    v["day"] = (v.visit_date - T0).dt.days
    tx["s"], tx["e"] = (tx.start_date - T0).dt.days, (tx.end_date - T0).dt.days
    last = v.groupby("patient_id").day.max()
    tx["e"] = tx.e.fillna(tx.patient_id.map(last))  # open courses: to the last visit
    ab = {p: np.sort(g.s.to_numpy()) for p, g in tx[tx.treatment_class == "Antibiotic"].groupby("patient_id")}
    bio = {p: union(zip(g.s, g.e)) for p, g in tx[tx.treatment_class == "Biologic"].groupby("patient_id")}
    print("visit types: " + ", ".join(f"{k} {n} ({v.ihs4_score[v.visit_type == k].notna().mean():.0%} scored)"
                                      for k, n in v.visit_type.value_counts().items()))

    rates, n2, gaps, cvs, amp, rat, dec, ab_lag, onoff, amp_p, n_pr, main_pr, rep = [], [], [], [], [], [], [], [], [], [], [], [], []
    for pid, g in v.groupby("patient_id", sort=False):
        d, ty, sc = g.day.to_numpy(), g.visit_type.to_numpy(), g.ihs4_score.to_numpy(float)
        pr = g.practitioner_id.to_numpy()
        n_pr.append(len(set(pr)))
        main_pr.append(pd.Series(pr).value_counts().iloc[0] / len(pr))
        fday = d[ty == "flare_reactive"]
        ss = np.flatnonzero(np.isin(ty, ("background", "treatment_monitoring")) & np.isfinite(sc))
        for a, b in zip(ss[:-1], ss[1:]):
            if d[b] - d[a] <= 21 and not np.any((fday >= d[a] - 90) & (fday <= d[b])):
                rep.append((abs(np.log1p(sc[b]) - np.log1p(sc[a])), pr[a] == pr[b], d[b] - d[a], np.log1p(sc[a])))
        span = (d[-1] - d[0]) / YEAR
        fl = np.flatnonzero(ty == "flare_reactive")
        if span >= 1:
            rates.append(len(fl) / span)
        if span >= 2:
            n2.append(((ty == "flare_reactive") & (d < d[0] + 2 * YEAR)).sum())
        if len(fl) >= 2:
            gaps += list(np.diff(d[fl]))
        if len(fl) >= 5:
            gg = np.diff(d[fl])
            cvs.append(gg.std() / gg.mean())
        bgi = np.flatnonzero((ty == "background") & np.isfinite(sc))
        for f in fl:
            if not np.isfinite(sc[f]):
                continue
            prev = bgi[(d[bgi] < d[f]) & (d[bgi] >= d[f] - 180)]
            if not len(prev):
                continue
            b, F = sc[prev[-1]], sc[f]
            amp.append(F - b)
            amp_p.append(pid)
            rat.append((1 + F) / (1 + b))
            if F - b >= 3:
                nxt = d[fl][d[fl] > d[f]]
                stop = nxt[0] if len(nxt) else np.inf
                for k in np.flatnonzero((d > d[f]) & (d <= d[f] + 180) & (d < stop) & np.isfinite(sc)):
                    dec.append((d[k] - d[f], (sc[k] - b) / (F - b), ty[k]))
        if pid in ab and len(fl):
            for f in fl:
                lag = ab[pid] - d[f]
                ab_lag.append(lag[np.argmin(np.abs(lag))])
        if pid in bio:
            on = sum(max(0, min(e, d[-1]) - max(s, d[0])) for s, e in bio[pid])
            off = d[-1] - d[0] - on
            isin = lambda x: any(s <= x <= e for s, e in bio[pid])
            fo = sum(isin(x) for x in d[fl])
            bg = np.flatnonzero(ty == "background")
            bo = sum(isin(x) for x in d[bg])
            if on >= 180 and off >= 180:
                onoff.append((fo, len(fl) - fo, bo, len(bg) - bo, on / YEAR, off / YEAR))

    r, n2 = np.array(rates), np.array(n2)
    print(f"\n1. flare visits per patient-year (span >= 1 year, n={len(r)}): mean {r.mean():.2f}, median {np.median(r):.2f}, "
          f"quartiles {np.percentile(r, 25):.2f} / {np.percentile(r, 75):.2f}, 90th {np.percentile(r, 90):.2f}; no flare visit {np.mean(r == 0):.0%}")
    print(f"   counts in the first 2 years (n={len(n2)}): mean {n2.mean():.2f}, var {n2.var():.2f}, var / mean {n2.var() / n2.mean():.2f} "
          f"(Poisson 1)")
    gaps = np.array(gaps)
    print(f"2. gaps between flare visits (n={len(gaps)}): median {np.median(gaps):.0f} days; < 30 days {np.mean(gaps < 30):.0%}, < 60 days "
          f"{np.mean(gaps < 60):.0%}; CV within patients with >= 5 flares (n={len(cvs)}): median {np.median(cvs):.2f} (exponential 1)")
    amp, rat = np.array(amp), np.array(rat)
    print(f"3. amplitude (n={len(amp)}): flare score - baseline median {np.median(amp):+.1f}, quartiles {np.percentile(amp, 25):+.1f} / "
          f"{np.percentile(amp, 75):+.1f}, 90th {np.percentile(amp, 90):+.1f}; <= 0 {np.mean(amp <= 0):.0%}; (1 + flare) / (1 + baseline) "
          f"median x{np.median(rat):.2f}")
    # do patients differ in amplitude? variance components of log((1 + flare) / (1 + baseline)) over patients with >= 3 flares
    la = pd.DataFrame({"p": amp_p, "a": np.log(rat)})
    g = la.groupby("p").a.agg(["mean", "var", "count"])
    g = g[g["count"] >= 3]
    within = g["var"].mean()
    between = g["mean"].var() - (g["var"] / g["count"]).mean()
    print(f"   patients differ? log amplitude ratio, {len(g)} patients with >= 3 flares: sd between patients {np.sqrt(max(between, 0)):.3f}, "
          f"within a patient {np.sqrt(within):.3f}; ICC {max(between, 0) / (max(between, 0) + within):.3f}")
    dd = pd.DataFrame(dec, columns=["days", "excess", "type"])
    dd["bin"] = pd.cut(dd.days, [0, 10, 20, 35, 60, 90, 180])
    print("4. decay of the excess after a flare (amplitude >= 3; 1 = still at the flare score, 0 = back at the baseline):")
    for b, g in dd.groupby("bin", observed=True):
        print(f"   days {str(b):10s} n={len(g):6d}  median {g.excess.median():+.2f}  mean {g.excess.mean():+.2f}  "
              f"(types: {', '.join(f'{k} {n}' for k, n in g.type.value_counts().items())})")
    lag = np.array(ab_lag)
    print(f"5. nearest antibiotic start to a flare visit (n={len(lag)}): within -7..+7 days {np.mean(np.abs(lag) <= 7):.0%}, "
          f"within 0..+14 {np.mean((lag >= 0) & (lag <= 14)):.0%}, -14..-1 {np.mean((lag >= -14) & (lag < 0)):.0%}; median |lag| "
          f"{np.median(np.abs(lag)):.0f} days")
    n_pr, main_pr = np.array(n_pr), np.array(main_pr)
    print(f"7. practitioners per patient: 1 {np.mean(n_pr == 1):.0%}, 2 {np.mean(n_pr == 2):.0%}, >= 3 {np.mean(n_pr >= 3):.0%}; share of the "
          f"visits with the main practitioner median {np.median(main_pr):.2f}; practitioners {v.practitioner_id.nunique()}")
    r = pd.DataFrame(rep, columns=["dy", "same", "gap", "y0"])
    for name, s in (("all", np.ones(len(r), bool)), ("same practitioner", r.same.to_numpy()), ("other practitioner", ~r.same.to_numpy()),
                    ("same, gap <= 7 days", r.same.to_numpy() & (r.gap <= 7).to_numpy())):
        x = r.dy[s]
        if len(x):
            print(f"   near-replicate pairs, {name:22s} n={len(x):6d}: mean |change| {x.mean():.3f} -> Laplace scale {x.mean() / 1.5:.3f} "
                  f"(sd {x.mean() / 1.5 * np.sqrt(2):.3f}); median gap {r.gap[s].median():.0f} days")
    if (~r.same).sum() > 30:
        vs, vo = (r.dy[r.same] ** 2).mean() / 2, (r.dy[~r.same] ** 2).mean() / 2
        print(f"   rater bias: sd {np.sqrt(max(vo - vs, 0) / 2):.3f} (from the mean squared changes, other - same practitioner)")
    o = np.array(onoff, float)
    print(f"6. biologics, {len(o)} patients with >= 180 days on and off: flare visits per year on {o[:, 0].sum() / o[:, 4].sum():.2f}, "
          f"off {o[:, 1].sum() / o[:, 5].sum():.2f}; background visits per year on {o[:, 2].sum() / o[:, 4].sum():.2f}, "
          f"off {o[:, 3].sum() / o[:, 5].sum():.2f}")


if __name__ == "__main__":
    main()
