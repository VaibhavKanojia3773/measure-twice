"""Do calibrated error bars survive physical degradation? Measured answers on 25 ScanNet++ scenes under low light, defocus
and H.264 compression, against the same scenes clean.

Calibration is frozen on CLEAN scenes disjoint from the 25 test scenes (one question per calibration scene, exact order
statistic, sigma0 from separate clean development scenes; 100 random draws). Uncertainty = spread of all available runs of
the given families (questions need >= 2 runs). Reported per task and condition: answer availability, full-denominator
score, interval coverage and median width, median spread, and whether the spread rises where the error rises.

python -m measure_twice.degradation_eval --clean results/vsi_answers.json --degraded results/vsi_answers_degraded.json --out results/degrade_geo.json
"""
import argparse, json, math
import numpy as np
from scipy.stats import spearmanr
from measure_twice.calibration import NUM, mra

TASKS = ("object_abs_distance", "object_size_estimation", "room_size_estimation", "object_counting")


def runs(r, fams):
    return [x for k, x in r["runs"].items() if k.startswith(fams) and x is not None and np.isfinite(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True); ap.add_argument("--degraded", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="mapanything,pi3x")
    a = ap.parse_args(); fams = tuple(f + "_" for f in a.families.split(","))
    C = [r for r in json.load(open(a.clean)) if r.get("cond", "clean") == "clean"]
    D = json.load(open(a.degraded))
    test_scenes = {(r["ds"], r["scene"]) for r in D}
    conds = ["clean"] + sorted({r["cond"] for r in D})
    rows = [dict(r, cond="clean") for r in C if (r["ds"], r["scene"]) in test_scenes] + D
    res = {}
    for t in TASKS:
        eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
        pool = {}
        for r in C:
            if r["type"] == t and (r["ds"], r["scene"]) not in test_scenes:
                v = runs(r, fams)
                if len(v) >= 2:
                    pool.setdefault(r["ds"] + "/" + r["scene"], []).append((float(np.median(v)), float(np.std(g(v))), float(r["gt"])))
        scenes = sorted(pool); rng = np.random.default_rng(0); Q = []
        for _ in range(100):
            p = rng.permutation(len(scenes)); dev, cal = p[: len(p) // 3], p[len(p) // 3:]
            s0 = float(np.median([x[1] for i in dev for x in pool[scenes[i]]]))
            sc = np.sort([abs(g(y) - g(pt)) / (s + s0) for i in cal for pt, s, y in [pool[scenes[i]][rng.integers(len(pool[scenes[i]]))]]])
            k = math.ceil((len(sc) + 1) * 0.9); Q.append((sc[k - 1] if k <= len(sc) else np.inf, s0))
        T = [r for r in rows if r["type"] == t]; ids = sorted({r["id"] for r in T})
        by = {(r["id"], r["cond"]): r for r in T}
        o = {}
        for c in conds:
            R = [by[(i, c)] for i in ids if (i, c) in by]
            full, cov, wid, sig, err = [], [], [], [], []
            for r in R:
                v = runs(r, fams); y = float(r["gt"])
                full.append(float(mra(np.array([np.median(v)]), np.array([y]))[0]) if v else 0.0)
                if len(v) >= 2:
                    pt, s = float(np.median(v)), float(np.std(g(v)))
                    lo = [max(np.exp(g(pt) - q * (s + s0)) - eps, 0) for q, s0 in Q]; hi = [np.exp(g(pt) + q * (s + s0)) - eps for q, s0 in Q]
                    cov.append(np.mean([l <= y <= h for l, h in zip(lo, hi)])); wid.append(np.median(np.array(hi) - np.array(lo)))
                    sig.append(s); err.append(abs(g(y) - g(pt)))
            o[c] = dict(n=len(R), avail=float(np.mean([len(runs(r, fams)) > 0 for r in R])), spread_avail=len(sig) / max(len(R), 1),
                        score=float(np.mean(full)), cov=float(np.mean(cov)) if cov else None, width=float(np.median(wid)) if wid else None,
                        sigma=float(np.median(sig)) if sig else None, err=float(np.median(err)) if err else None,
                        rho=float(spearmanr(sig, err).correlation) if len(sig) > 5 else None)
        # does the spread rise where the error rises? paired over questions with a spread in both conditions
        for c in conds[1:]:
            ds_, de_ = [], []
            for i in ids:
                if (i, "clean") in by and (i, c) in by:
                    v0, v1 = runs(by[(i, "clean")], fams), runs(by[(i, c)], fams)
                    if len(v0) >= 2 and len(v1) >= 2:
                        y = float(by[(i, c)]["gt"])
                        ds_.append(np.std(g(v1)) - np.std(g(v0))); de_.append(abs(g(y) - g(np.median(v1))) - abs(g(y) - g(np.median(v0))))
            o[c]["paired_n"] = len(ds_); o[c]["dsigma_derr_rho"] = float(spearmanr(ds_, de_).correlation) if len(ds_) > 5 else None
        res[t] = o
    for t in ("object_rel_distance", "object_rel_direction", "obj_appearance_order"):
        T = [r for r in rows if r["type"].startswith(t)]; o = {}
        for c in conds:
            sc = []
            for r in [x for x in T if x["cond"] == c]:
                v = [x for k, x in r["runs"].items() if k.startswith(fams) and x is not None]
                if not v:
                    sc.append(0.0); continue
                vals, cnt = np.unique(v, return_counts=True); sc.append(float(vals[np.argmax(cnt)] == r["gt"]))
            o[c] = dict(n=len(sc), score=float(np.mean(sc)) if sc else None)
        res[t] = o
    json.dump(res, open(a.out, "w"), indent=1)
    for t, o in res.items():
        print(t)
        for c, v in o.items():
            print("   ", f"{c:10s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
