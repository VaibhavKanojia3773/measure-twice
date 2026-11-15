"""Robustness analyses added for the camera-ready version (results/analysis/robustness.json). CPU only, from saved run answers.
All calibration uses measure_twice.calibration.conformal / selective unchanged (same splits, sigma0 rule, one question per calibration
scene, alpha = 0.1).

  population   Complete-output subset. (a) composition of complete vs. answered-but-incomplete questions;
               (b) the same uncertainty comparison on the larger population with ANY >= 2 finite runs of the two families
               (point = median of the available runs, spread = std of the available runs), and on MapAnything alone
               (its three runs exist for far more questions, including the long ARKitScenes videos).
  widths       How wide are the intervals? Multiplicative half-width factor F = exp(q (sigma + sigma0)), i.e. the
               interval is about [y/F, y*F]; width percentiles; width on the most-certain half.
  ownpoint     Three-run ablation with each row's own answer. Each three-run signal is evaluated with the
               median of ITS OWN three runs as the answer: ranking (Spearman, AURC, gain over random order) and conformal
               intervals; paired scene bootstrap of the ranking-gain difference within-model minus across-model.

python scripts/robustness.py
"""
import json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from measure_twice import calibration as V
from measure_twice.calibration import NUM, mra, finite, table, conformal, selective

A = json.load(open(os.path.join(ROOT, "results", "vsi_answers.json")))
TASKS = ("object_abs_distance", "object_size_estimation", "room_size_estimation")
F2 = ("mapanything", "pi3x"); F3 = ("mapanything", "pi3x", "da3m")
out = {}


def any_table(rows, t, fams, min_runs=2):
    """Questions with >= min_runs finite runs over the families (no completeness requirement)."""
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps); T = []
    for r in rows:
        v = sum((finite(r, f) for f in fams), [])
        if len(v) < min_runs:
            continue
        T.append(dict(id=r["id"], ds=r["ds"], scene=r["ds"] + "/" + r["scene"], gt=float(r["gt"]), point=float(np.median(v)),
                      n_runs=len(v), sig={"const": 1.0, "comb": float(np.std(g(v)))}))
    return T


def width_factors(T, t, sig, alpha=0.10, seed=0, point="point"):
    """Replicates the split loop of vsi_core.conformal and records multiplicative half-width factors."""
    eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
    scenes = sorted({r["scene"] for r in T}); by = {s: [r for r in T if r["scene"] == s] for s in scenes}
    rng = np.random.default_rng(seed); Fm, W10, W50, W90, Whalf, Fhalf = [], [], [], [], [], []
    for _ in range(V.SPLITS):
        p = rng.permutation(len(scenes)); n = len(scenes)
        dev, cal, te = [scenes[i] for i in p[: n // 5]], [scenes[i] for i in p[n // 5: 3 * n // 5]], [scenes[i] for i in p[3 * n // 5:]]
        SG = V.sigma_map(T, sig, set(dev)); s0 = float(np.median([SG[r["id"]] for s in dev for r in by[s]])) or 1e-3
        C = [by[s][rng.integers(len(by[s]))] for s in cal]
        sc = np.sort([abs(g(r["gt"]) - g(r[point])) / (SG[r["id"]] + s0) for r in C]); k = math.ceil((len(sc) + 1) * (1 - alpha))
        if k > len(sc):
            continue
        q = sc[k - 1]; R = [r for s in te for r in by[s]]
        u = np.array([SG[r["id"]] for r in R]); f = np.exp(q * (u + s0))
        lo = np.maximum(np.exp(g(np.array([r[point] for r in R])) - np.log(f)) - eps, 0)
        hi = np.exp(g(np.array([r[point] for r in R])) + np.log(f)) - eps; w = hi - lo
        half = u <= np.median(u)
        Fm.append(np.median(f)); W10.append(np.percentile(w, 10)); W50.append(np.median(w)); W90.append(np.percentile(w, 90))
        Whalf.append(np.median(w[half])); Fhalf.append(np.median(f[half]))
    m = lambda x: float(np.mean(x))
    return dict(factor=m(Fm), w10=m(W10), w50=m(W50), w90=m(W90), w_certain_half=m(Whalf), factor_certain_half=m(Fhalf))


def gain_boot(T, t, s1, p1, s2, p2, B=1000):
    """Paired scene bootstrap of [random-AURC - AURC] for (signal s1, answer p1) minus (s2, p2); positive = s1 ranks better."""
    y = np.array([r["gt"] for r in T]); m1 = mra(np.array([r[p1] for r in T]), y); m2 = mra(np.array([r[p2] for r in T]), y)
    a, b = np.array([r["sig"][s1] for r in T]), np.array([r["sig"][s2] for r in T])
    scenes = sorted({r["scene"] for r in T}); idx = {s: [i for i, r in enumerate(T) if r["scene"] == s] for s in scenes}
    rng = np.random.default_rng(0); tb = rng.random(len(T))

    def gain(ii, m, s):
        o = np.lexsort((tb[ii], s[ii])); c = np.cumsum(m[ii][o]) / np.arange(1, len(ii) + 1); return (1 - m[ii].mean()) - (1 - c.mean())
    d = [gain(ii, m1, a) - gain(ii, m2, b) for ii in
         (np.concatenate([idx[s] for s in rng.choice(scenes, len(scenes))]) for _ in range(B))]
    return dict(mean=float(np.mean(d)), p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)))


for t in TASKS:
    R = [r for r in A if r["type"] == t and r.get("cond", "clean") == "clean"]
    o = {}
    # ---------- population
    Tc = table(R, t, list(F2)); cid = {r["id"] for r in Tc}
    Ta = any_table(R, t, F2)
    inc = [r for r in Ta if r["id"] not in cid]
    comp = lambda S: dict(n=len(S), n_scenes=len({r["scene"] for r in S}),
                          mra=float(np.mean(mra(np.array([r["point"] for r in S]), np.array([r["gt"] for r in S])))) if S else None,
                          median_gt=float(np.median([r["gt"] for r in S])) if S else None,
                          ds={d: sum(r["ds"] == d for r in S) for d in ("arkitscenes", "scannet", "scannetpp")})
    o["population"] = dict(n_all=len(R), complete=comp([dict(r) for r in Tc]), incomplete_answered=comp(inc),
                           any2=comp(Ta), n_pred=sum(bool(sum((finite(r, f) for f in F2), [])) for r in R))
    o["any2"] = {s: dict(conf=conformal(Ta, t, s), sel=selective(Ta, t, s)) for s in ("const", "comb")}
    Tm = table(R, t, ["mapanything"])
    o["ma_only"] = dict(n=len(Tm), n_scenes=len({r["scene"] for r in Tm}),
                        **{s: dict(conf=conformal(Tm, t, s), sel=selective(Tm, t, s)) for s in ("const", "comb")})
    # ---------- widths (main common set, same as Table 1)
    o["widths"] = {s: width_factors(Tc, t, s) for s in ("const", "comb", "native:mean")}
    # ---------- own-point three-run ablation (three-family common set, same as Table 2)
    T3 = table(R, t, list(F3))
    for r, raw in zip(T3, [next(x for x in R if x["id"] == r["id"]) for r in T3]):
        for f in F3:
            r[f"point_{f}"] = float(np.median(finite(raw, f)))
    rows = {f"within:{f}": (f"fam:{f}", f"point_{f}") for f in F3}; rows["across"] = ("budget:across", "point_across")
    rows["all9"] = ("comb", "point")
    own = {}
    for name, (s, p) in rows.items():
        se = selective(T3, t, s, point=p); c = conformal(T3, t, s, point=p)
        own[name] = dict(rho=se["spearman"], aurc=se["aurc"], aurc_random=se["aurc_random"], gain=se["aurc_random"] - se["aurc"],
                         mra=se["mra"], cov=c["cov"], width=c["width"])
    own["boot_gain_within_minus_across"] = {f: gain_boot(T3, t, f"fam:{f}", f"point_{f}", "budget:across", "point_across") for f in F3}
    o["ownpoint"] = dict(n=len(T3), rows=own)
    out[t] = o
    P = o["population"]
    print(f"\n== {t}: all {P['n_all']} | answered {P['n_pred']} | >=2 runs {P['any2']['n']} | complete {P['complete']['n']} "
          f"| incomplete-answered {P['incomplete_answered']['n']}")
    print(f"   point MRA complete {P['complete']['mra']:.3f}  incomplete {P['incomplete_answered']['mra'] if P['incomplete_answered']['mra'] is not None else float('nan'):.3f}"
          f"  ds complete {P['complete']['ds']}  ds incomplete {P['incomplete_answered']['ds']}")
    for k in ("any2",):
        for s in ("const", "comb"):
            c, se = o[k][s]["conf"], o[k][s]["sel"]
            print(f"   {k:6s} {s:6s} cov {100 * c['cov']:.1f} [{100 * c['cov_p05']:.1f},{100 * c['cov_p95']:.1f}] width {c['width']:.2f} rho {se['spearman']:.2f} AURC {se['aurc']:.3f}")
    for s in ("const", "comb"):
        c, se = o["ma_only"][s]["conf"], o["ma_only"][s]["sel"]
        print(f"   MAonly n={o['ma_only']['n']} {s:6s} cov {100 * c['cov']:.1f} width {c['width']:.2f} rho {se['spearman']:.2f} AURC {se['aurc']:.3f}")
    for s, w in o["widths"].items():
        print(f"   widths {s:12s} factor x{w['factor']:.2f}  w10/50/90 {w['w10']:.2f}/{w['w50']:.2f}/{w['w90']:.2f}  certain-half width {w['w_certain_half']:.2f} (x{w['factor_certain_half']:.2f})")
    print(f"   own-point ablation (n={len(T3)}):")
    for name, v in own.items():
        if name.startswith("boot"):
            print("     boot gain within-across", {f: (round(x['mean'], 3), round(x['p05'], 3), round(x['p95'], 3)) for f, x in v.items()})
        else:
            print(f"     {name:20s} rho {v['rho']:.2f} AURC {v['aurc']:.3f} rand {v['aurc_random']:.3f} gain {v['gain']:.3f} MRA {v['mra']:.3f} cov {100 * v['cov']:.1f} width {v['width']:.2f}")
os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "analysis", "robustness.json"), "w"), indent=1)
