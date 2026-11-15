"""Canonical uncertainty analysis for metric VSI-Bench answers. Every number, figure and transfer result in the paper comes
from this file, so all of them evaluate the same predictor and the same uncertainty scores.

Per question (numeric tasks), with g(x) = log(x + eps):
  point     median of the finite run answers over the chosen families (held fixed when uncertainty signals are compared)
  signals   const        no uncertainty (constant-width conformal)
            fam:<f>      spread within family f: std of g over its frame-subset runs
            between      spread between families: std of the family medians in g
            comb         spread over all runs
            budget:*     matched-budget pairs: 3 runs of one family vs. one full-video run of each of 3 families
Calibration protocol (repeated SPLITS times):
  scenes are split into development / calibration / test (20 / 40 / 40 %); questions of a scene never cross a split.
  sigma0 (score stabiliser) is the median signal on development scenes only.
  calibration uses ONE random question per calibration scene, so calibration scores are exchangeable with the score of a
  random question from a random unseen scene; q = the ceil((n+1)(1-alpha))-th smallest score (inf if that rank exceeds n).
  Test coverage is averaged per scene first (target: a random question of a random unseen scene).

python -m measure_twice.calibration --answers results/vsi_answers.json --families mapanything,pi3x --out results/analysis_2fam.json
"""
import argparse, json, math
import numpy as np
from scipy.stats import spearmanr

NUM = {"object_counting": 1.0, "object_abs_distance": 1e-2, "object_size_estimation": 1.0, "room_size_estimation": 0.1}
UNIT = {"object_abs_distance": "m", "object_size_estimation": "cm", "room_size_estimation": "m2", "object_counting": ""}
SPLITS, Z90 = 100, 1.645


def mra(pred, gt):
    rel = np.abs(pred - gt) / np.maximum(np.abs(gt), 1e-9)
    return np.mean([rel < 1 - th for th in np.arange(0.5, 1.0, 0.05)], 0)


def finite(r, fam):
    return [v for k, v in sorted(r["runs"].items()) if k.startswith(fam + "_") and v is not None and np.isfinite(v)]


def table(rows, t, fams):
    """One record per question with every run present for every family (the common evaluation set)."""
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
    out = []
    for r in rows:
        per = {f: finite(r, f) for f in fams}
        if any(len(v) < 3 for v in per.values()):
            continue
        allv = sum(per.values(), [])
        rec = dict(id=r["id"], ds=r["ds"], scene=r["ds"] + "/" + r["scene"], gt=float(r["gt"]), point=float(np.median(allv)))
        S = {"const": 1.0, "comb": float(np.std(g(allv))), "between": float(np.std([np.median(g(v)) for v in per.values()]))}
        for f, v in per.items():
            S[f"fam:{f}"] = float(np.std(g(v)))
        if len(fams) >= 3:                     # matched budget: three runs each
            for f in fams:
                S[f"budget:within:{f}"] = S[f"fam:{f}"]
            full = [r["runs"][f"{f}_all"] for f in fams]
            S["budget:across"] = float(np.std(g(full)))
            rec["point_across"] = float(np.median(full))
        rec["sig"] = S
        rec["nat"] = {f: (r.get("native") or {}).get(f"{f}_all") for f in fams}
        dets = [v for v in (r.get("det") or {}).values() if v is not None]
        rec["detc"] = float(np.mean(dets)) if dets else None
        out.append(rec)
    # confidence baselines: native 3D confidence and detector score are kept raw. Ranking metrics (AURC, Spearman) use a
    # tie-aware rank over the evaluated questions (label-free, order only); calibrated intervals instead map confidence to
    # an uncertainty with an empirical CDF fitted on development scenes only (see sigma_map), so the score is fixed before
    # calibration and test questions are seen.
    from scipy.stats import rankdata

    def rank_unc(vals):
        v = np.array([np.nan if x is None else x for x in vals], float); ok = np.isfinite(v)
        u = np.full(len(v), 1.0); u[ok] = 1 - (rankdata(v[ok]) - 0.5) / ok.sum()
        return u
    for rec in out:
        rec["raw"] = {f"native:{f}": rec["nat"][f] for f in fams}; rec["raw"]["detector"] = rec["detc"]
    if out and all(any(rec["nat"][f] is not None for rec in out) for f in fams):
        U = {f: rank_unc([rec["nat"][f] for rec in out]) for f in fams}
        for i, rec in enumerate(out):
            for f in fams:
                rec["sig"][f"native:{f}"] = float(U[f][i])
            rec["sig"]["native:mean"] = float(np.mean([U[f][i] for f in fams]))
    if out and any(rec["detc"] is not None for rec in out):
        D = rank_unc([rec["detc"] for rec in out])
        for i, rec in enumerate(out):
            rec["sig"]["detector"] = float(D[i])
    return out


CONF_SIGNALS = ("native:", "detector")


def sigma_map(T, sig, dev):
    """Uncertainty of every record for one split. Spread signals are fixed functions of the runs. Confidence signals are
    mapped through 1 - ECDF of the development-scene confidences (mid-rank for ties); missing confidence -> 1."""
    if not sig.startswith(CONF_SIGNALS):
        return {r["id"]: r["sig"][sig] for r in T}
    if sig == "native:mean":
        parts = sorted({k for r in T for k in r["raw"] if k.startswith("native:")})
    else:
        parts = [sig]
    out = {r["id"]: [] for r in T}
    for p in parts:
        ref = np.sort([r["raw"][p] for r in T if r["scene"] in dev and r["raw"].get(p) is not None])
        for r in T:
            x = r["raw"].get(p)
            if x is None or not len(ref):
                out[r["id"]].append(1.0)
            else:
                out[r["id"]].append(1 - (np.searchsorted(ref, x, "left") + np.searchsorted(ref, x, "right")) / (2 * len(ref)))
    return {i: float(np.mean(v)) for i, v in out.items()}


def raw_cov(T, t, sig):
    """Coverage of the uncalibrated Gaussian interval g(point) +- 1.645 * spread (a heuristic, not a model output)."""
    eps = NUM[t]; g = lambda x: np.log(x + eps)
    return float(np.mean([abs(g(r["gt"]) - g(r["point"])) <= Z90 * r["sig"][sig] for r in T]))


def conformal(T, t, sig, alpha=0.10, seed=0, point="point", thresholds=(), keep=None):
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps); gi = lambda y: np.exp(y) - eps
    scenes = sorted({r["scene"] for r in T}); by = {s: [r for r in T if r["scene"] == s] for s in scenes}
    rng = np.random.default_rng(seed)
    acc = dict(cov=[], cov_q=[], width=[], width_p90=[], relwidth=[], iscore=[], is_mean=[], inf=0, dec_frac=[], dec_err=[],
               pt_err=[], ctl_err=[])
    for _ in range(SPLITS):
        p = rng.permutation(len(scenes)); n = len(scenes)
        dev, cal, te = [scenes[i] for i in p[: n // 5]], [scenes[i] for i in p[n // 5: 3 * n // 5]], [scenes[i] for i in p[3 * n // 5:]]
        SG = sigma_map(T, sig, set(dev))
        s0 = float(np.median([SG[r["id"]] for s in dev for r in by[s]])) or 1e-3
        C = [by[s][rng.integers(len(by[s]))] for s in cal]
        sc = np.sort([abs(g(r["gt"]) - g(r[point])) / (SG[r["id"]] + s0) for r in C])
        k = math.ceil((len(sc) + 1) * (1 - alpha))
        q = sc[k - 1] if k <= len(sc) else np.inf
        if not np.isfinite(q):
            acc["inf"] += 1; continue
        if keep is not None:                    # expose each split (for the figure that instantiates one of them)
            keep.append(dict(dev=set(dev), test=set(te), s0=s0, q=float(q), sigma=SG))
        cs, ws, rw, iss, ism, qs = [], [], [], [], [], []
        df, de, pe, mg = [], [], [], []
        for s in te:
            R = by[s]
            lo = np.array([max(gi(g(r[point]) - q * (SG[r["id"]] + s0)), 0.0) for r in R])
            hi = np.array([gi(g(r[point]) + q * (SG[r["id"]] + s0)) for r in R])
            y = np.array([r["gt"] for r in R]); pt = np.array([r[point] for r in R])
            inside = (lo <= y) & (y <= hi)
            cs.append(inside.mean()); qs += inside.tolist(); ws += (hi - lo).tolist(); rw += ((hi - lo) / y).tolist()
            isa = (hi - lo) + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)
            iss += (isa / y).tolist(); ism += isa.tolist()
            for tau in thresholds:             # "is the quantity larger than tau?" decided only if the interval excludes tau
                d = (lo > tau) | (hi < tau); truth = y > tau
                df += d.tolist(); de += ((lo > tau) != truth)[d].tolist(); pe += ((pt > tau) != truth).tolist()
                mg += np.abs(g(pt) - g(tau)).tolist()      # control: decide the pairs whose point answer is farthest from tau
        acc["cov"].append(np.mean(cs)); acc["cov_q"].append(np.mean(qs)); acc["width"].append(np.median(ws))
        acc["width_p90"].append(np.percentile(ws, 90)); acc["is_mean"].append(np.mean(ism))
        acc["relwidth"].append(np.median(rw)); acc["iscore"].append(np.median(iss))   # median per-question relative interval score
        if thresholds:
            acc["dec_frac"].append(np.mean(df)); acc["dec_err"].append(np.mean(de) if de else np.nan); acc["pt_err"].append(np.mean(pe))
            k = int(np.sum(df)); o_ = np.lexsort((rng.random(len(mg)), -np.array(mg)))[:k]
            acc["ctl_err"].append(float(np.mean(np.array(pe)[o_])) if k else np.nan)
    o = {k: (float(np.nanmean(v)) if isinstance(v, list) and v else v) for k, v in acc.items()}
    o["cov_p05"], o["cov_p95"] = (float(np.percentile(acc["cov"], 5)), float(np.percentile(acc["cov"], 95))) if acc["cov"] else (None, None)
    o["n_q"], o["n_scenes"] = len(T), len(scenes)
    return o


def selective(T, t, sig, point="point"):
    m = mra(np.array([r[point] for r in T]), np.array([r["gt"] for r in T]))
    s = np.array([r["sig"][sig] for r in T]); res = np.abs(np.log(np.array([r["gt"] for r in T]) + NUM[t]) - np.log(np.array([r[point] for r in T]) + NUM[t]))
    rng = np.random.default_rng(0); au, at = [], {25: [], 50: [], 75: []}
    for _ in range(20):                         # ties broken at random, averaged
        o = np.lexsort((rng.random(len(s)), s)); c = np.cumsum(m[o]) / np.arange(1, len(m) + 1)
        au.append(1 - c.mean())
        for f in at:
            at[f].append(c[max(1, int(f / 100 * len(m))) - 1])
    rho = float(spearmanr(s, res).correlation) if np.std(s) > 0 else 0.0
    return dict(aurc=float(np.mean(au)), aurc_random=float(1 - m.mean()), mra=float(m.mean()), spearman=rho,
                **{f"mra_at_{f}": float(np.mean(v)) for f, v in at.items()})


def boot_aurc(T, t, s1, s2, B=1000, point="point"):
    """Paired scene bootstrap of AURC(s1) - AURC(s2) on the same questions (negative = s1 better)."""
    m = mra(np.array([r[point] for r in T]), np.array([r["gt"] for r in T])); scenes = sorted({r["scene"] for r in T})
    idx = {s: [i for i, r in enumerate(T) if r["scene"] == s] for s in scenes}; rng = np.random.default_rng(0)
    a, b = np.array([r["sig"][s1] for r in T]), np.array([r["sig"][s2] for r in T]); tb = rng.random(len(T))

    def au(ii, s):
        o = np.lexsort((tb[ii], s[ii])); c = np.cumsum(m[ii][o]) / np.arange(1, len(ii) + 1); return 1 - c.mean()
    d = []
    for _ in range(B):
        ii = np.concatenate([idx[s] for s in rng.choice(scenes, len(scenes))]); d.append(au(ii, a) - au(ii, b))
    return dict(mean=float(np.mean(d)), p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)))


def full_denominator(rows, t, fams):
    """Selective answering over ALL questions of a task: questions without a spread estimate are answered last, questions
    without any prediction score 0 whenever they are 'answered'."""
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
    sc, key = [], []
    for r in rows:
        v = sum((finite(r, f) for f in fams), [])
        if not v:
            sc.append(0.0); key.append((2, 0.0))
        elif len(v) < 2:
            sc.append(float(mra(np.median(v), float(r["gt"])))); key.append((1, 0.0))
        else:
            sc.append(float(mra(np.median(v), float(r["gt"])))); key.append((0, float(np.std(g(v)))))
    o = sorted(range(len(sc)), key=lambda i: key[i]); s = np.array(sc)[o]
    c = np.cumsum(s) / np.arange(1, len(s) + 1)
    return dict(n=len(rows), n_pred=sum(k[0] < 2 for k in key), n_spread=sum(k[0] == 0 for k in key), mra_full=float(s.mean()),
                **{f"mra_at_{f}": float(c[int(f / 100 * len(s)) - 1]) for f in (25, 50, 75)},
                curve=[float(x) for x in c[:: max(1, len(c) // 200)]])


def choice(rows, fams):
    """Multiple choice over ALL questions of a task: majority vote of the runs, confidence = vote share (0 if no run
    answered, which then scores 0). Ties in confidence are broken at random and averaged."""
    acc, conf = [], []
    for r in rows:
        v = [x for k, x in r["runs"].items() if x is not None and any(k.startswith(f + "_") for f in fams)]
        n = sum(1 for k in r["runs"] if any(k.startswith(f + "_") for f in fams))
        if not v:
            acc.append(0.0); conf.append(0.0); continue
        vals, cnt = np.unique(v, return_counts=True)
        acc.append(float(vals[np.argmax(cnt)] == r["gt"])); conf.append(cnt.max() / max(n, 1))
    acc, conf = np.array(acc), np.array(conf); rng = np.random.default_rng(0); au, at = [], {25: [], 50: [], 75: []}
    for _ in range(20):
        o = np.lexsort((rng.random(len(conf)), -conf)); c = np.cumsum(acc[o]) / np.arange(1, len(acc) + 1)
        au.append(1 - c.mean())
        for f in at:
            at[f].append(c[max(1, int(f / 100 * len(acc))) - 1])
    return dict(n=len(rows), acc=float(acc.mean()), aurc=float(np.mean(au)), aurc_random=float(1 - acc.mean()),
                **{f"acc_at_{f}": float(np.mean(v)) for f, v in at.items()})


def transfer(T, t, sig="comb", alpha=0.10):
    """Calibrate on scenes of one source dataset (one question per scene), test on every other dataset separately."""
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
    out = {}
    for src in ("arkitscenes", "scannet", "scannetpp"):
        S = [r for r in T if r["ds"] == src]; scenes = sorted({r["scene"] for r in S})
        if len(scenes) < 10:
            continue
        rng = np.random.default_rng(0); cov = {d: [] for d in ("arkitscenes", "scannet", "scannetpp") if d != src}
        for _ in range(SPLITS):
            p = rng.permutation(len(scenes)); dev, cal = [scenes[i] for i in p[: len(p) // 3]], [scenes[i] for i in p[len(p) // 3:]]
            s0 = float(np.median([r["sig"][sig] for r in S if r["scene"] in dev]))
            C = [rng.choice([r for r in S if r["scene"] == s]) for s in cal]
            sc = np.sort([abs(g(r["gt"]) - g(r["point"])) / (r["sig"][sig] + s0) for r in C]); k = math.ceil((len(sc) + 1) * (1 - alpha))
            q = sc[k - 1] if k <= len(sc) else np.inf
            for d in cov:
                D = [r for r in T if r["ds"] == d]
                if not D:
                    continue
                ds_ = sorted({r["scene"] for r in D})
                cov[d].append(np.mean([np.mean([abs(g(r["gt"]) - g(r["point"])) <= q * (r["sig"][sig] + s0) for r in D if r["scene"] == s]) for s in ds_]))
        for d, v in cov.items():
            if v:
                out[f"{src}->{d}"] = float(np.mean(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True); ap.add_argument("--families", default="mapanything,pi3x"); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="object_abs_distance,object_size_estimation,room_size_estimation,object_counting")
    a = ap.parse_args(); fams = a.families.split(",")
    rows = json.load(open(a.answers)); res = {"families": fams}
    for t in a.tasks.split(","):
        R = [r for r in rows if r["type"] == t]; T = table(R, t, fams)
        sigs = list(T[0]["sig"])
        thr = (1.0, 2.0, 3.0) if t == "object_abs_distance" else ()
        o = dict(n_all=len(R), n_common=len(T), unit=UNIT[t], full=full_denominator(R, t, fams), signals={})
        for s in sigs:
            o["signals"][s] = dict(raw_cov90=raw_cov(T, t, s) if s in ("comb", "between") or s.startswith(("fam:", "budget:")) else None,
                                   conf=conformal(T, t, s, thresholds=thr), sel=selective(T, t, s))
        o["alpha"] = {str(al): conformal(T, t, "comb", alpha=al) for al in (0.2, 0.05)}
        o["alpha_const"] = {str(al): conformal(T, t, "const", alpha=al) for al in (0.2, 0.05)}
        if len(fams) >= 3:
            o["budget_system"] = {"across": dict(conf=conformal(T, t, "budget:across", point="point_across"),
                                                 sel=selective(T, t, "budget:across", point="point_across"))}
        o["transfer"] = transfer(T, t)
        o["boot"] = {f"comb-{b}": boot_aurc(T, t, "comb", b) for b in ("const", "detector", "native:mean", "between", "fam:mapanything", "fam:pi3x")
                     if b in T[0]["sig"]}
        if len(fams) >= 3:
            o["boot"].update({f"fam:{f}-budget:across": boot_aurc(T, t, f"fam:{f}", "budget:across") for f in fams})
        res[t] = o
        print(f"\n== {t}: all {len(R)}, common {len(T)} questions, {o['conf'] if False else ''}")
        print(f"   {'signal':24s} {'raw90':>6s} {'cov':>6s} {'p5-p95':>11s} {'width':>7s} {'relW':>5s} {'IS':>6s} {'AURC':>6s} {'rho':>6s} {'@50':>5s}")
        for s, v in o["signals"].items():
            c, se = v["conf"], v["sel"]; rc = f"{100 * v['raw_cov90']:.1f}" if v["raw_cov90"] is not None else "-"
            print(f"   {s:24s} {rc:>6s} {100 * c['cov']:6.1f} {100 * c['cov_p05']:5.1f}-{100 * c['cov_p95']:4.1f} {c['width']:7.2f} {c['relwidth']:5.2f} "
                  f"{c['iscore']:6.2f} {se['aurc']:6.3f} {se['spearman']:6.3f} {100 * se['mra_at_50']:5.1f}")
        if thr:
            for s in ("const", "comb", "between"):
                c = o["signals"][s]["conf"]
                print(f"   decision {s:8s}: decided {100 * c['dec_frac']:.1f}%  error among decided {100 * c['dec_err']:.1f}%  point-estimate error {100 * c['pt_err']:.1f}%  matched-coverage control {100 * c['ctl_err']:.1f}%")
        print("   transfer", {k: round(100 * v, 1) for k, v in o["transfer"].items()})
        print("   full denominator", {k: v for k, v in o["full"].items() if k != "curve"})
    for t, pref in (("object_rel_distance", "object_rel_distance"), ("object_rel_direction", "object_rel_direction"),
                    ("obj_appearance_order", "obj_appearance_order")):
        R = [r for r in rows if r["type"].startswith(pref)]
        if R:
            res[t] = choice(R, fams); print(t, {k: round(v, 3) for k, v in res[t].items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
