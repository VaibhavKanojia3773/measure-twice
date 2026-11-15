"""Spatial MLLMs next to the measured answers: accuracy, whether the MLLM's own confidence ranks its errors, and routing.

Confidence of the MLLM: probability of the chosen option (choices) or mean token log-likelihood of the number (numbers).
Measured answers and their uncertainty come from the canonical answers file (measure_twice.calibration conventions, given families).
Routing: per task, use the measured answer when its uncertainty is below a threshold, otherwise the MLLM. Thresholds are
chosen on development scenes and evaluated on disjoint test scenes (100 random 50/50 scene splits). Controls at the same
evaluation: MLLM only, measurement only (MLLM where no measurement exists), a per-task choice of the better source made
on development scenes, random routing at the same per-task rate, and an oracle that picks the better answer.

python -m measure_twice.mllm_compare --mllm results/mllm/qwen3vl8b_16f/*.jsonl --answers results/vsi_answers.json --out results/mllm_qwen.json
"""
import argparse, collections, glob, json
import numpy as np
from scipy.stats import spearmanr
from measure_twice.calibration import NUM, mra

CHOICE_PREFIX = ("object_rel_distance", "object_rel_direction", "obj_appearance_order")
TASK = lambda t: "object_rel_direction" if t.startswith("object_rel_direction") else t


def score(t, pred, gt):
    if pred is None:
        return 0.0
    if TASK(t) in NUM:
        try:
            return float(mra(np.array([float(pred)]), np.array([float(gt)]))[0])
        except (TypeError, ValueError):
            return 0.0
    return float(str(pred).strip().upper()[:1] == str(gt).strip().upper()[:1])


def aurc(s, conf, reps=20, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        o = np.lexsort((rng.random(len(conf)), -np.asarray(conf))); c = np.cumsum(np.asarray(s)[o]) / np.arange(1, len(s) + 1)
        out.append(1 - c.mean())
    return float(np.mean(out))


def measured(r, fams):
    """Measured answer, its score and an uncertainty (lower = more certain), or None if no run answered."""
    t = TASK(r["type"])
    v = [x for k, x in r["runs"].items() if x is not None and k.startswith(fams) and (not isinstance(x, float) or np.isfinite(x))]
    if not v:
        return None
    if t in NUM:
        p = float(np.median(v)); u = float(np.std(np.log(np.array(v) + NUM[t]))) if len(v) >= 2 else 9.0
        return p, score(t, p, r["gt"]), u
    vals, cnt = np.unique(v, return_counts=True); n = sum(1 for k in r["runs"] if k.startswith(fams))
    p = vals[np.argmax(cnt)]
    return p, score(t, p, r["gt"]), 1 - cnt.max() / n


def load_mllm(paths):
    M = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); M[r["id"]] = r
    return M


def mllm_conf(r):
    if r.get("probs") and TASK(r["type"]) not in NUM:
        return float(max(r["probs"].values()))
    return float(r.get("mean_logprob", -9.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mllm", nargs="+", required=True); ap.add_argument("--answers", required=True)
    ap.add_argument("--families", default="mapanything,pi3x"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); fams = tuple(f + "_" for f in a.families.split(","))
    M = load_mllm(sum((glob.glob(p) for p in a.mllm), []))
    A = {r["id"]: r for r in json.load(open(a.answers)) if r.get("cond", "clean") == "clean"}
    res = dict(n_mllm=len(M), tasks={})
    rows = []                                    # one per question answered by the MLLM (route planning excluded)
    for i, m in M.items():
        t = TASK(m["type"])
        if t == "route_planning":
            continue
        g = measured(A[i], fams) if i in A else None
        rows.append(dict(id=i, t=t, scene=(A[i]["ds"] + "/" + A[i]["scene"]) if i in A else f"?{i}",
                         ms=score(m["type"], m.get("pred"), m["gt"]), mc=mllm_conf(m),
                         gs=g[1] if g else 0.0, gu=g[2] if g else np.inf, has_g=g is not None))
    tasks = sorted({r["t"] for r in rows})
    for t in tasks:
        R = [r for r in rows if r["t"] == t]
        ms, mc = np.array([r["ms"] for r in R]), np.array([r["mc"] for r in R])
        o = dict(n=len(R), mllm=float(ms.mean()), geo=float(np.mean([r["gs"] for r in R])),
                 mllm_aurc=aurc(ms, mc), mllm_aurc_random=float(1 - ms.mean()),
                 mllm_rho=float(spearmanr(mc, -ms).correlation) if np.std(mc) > 0 else 0.0)
        if t not in NUM:
            o["mllm_mean_conf"] = float(mc.mean())          # stated confidence vs accuracy
        # matched subset: questions with a measured spread; each system ranked by its own confidence
        S = [r for r in R if r["has_g"] and np.isfinite(r["gu"]) and r["gu"] < 9]
        if len(S) >= 20:
            o["matched_n"] = len(S)
            o["matched_mllm_aurc"] = aurc([r["ms"] for r in S], [r["mc"] for r in S])
            o["matched_geo_aurc"] = aurc([r["gs"] for r in S], [-r["gu"] for r in S])
            o["matched_mllm_random"] = float(1 - np.mean([r["ms"] for r in S]))
            o["matched_geo_random"] = float(1 - np.mean([r["gs"] for r in S]))
        o["oracle"] = float(np.mean([max(r["ms"], r["gs"]) for r in R]))
        res["tasks"][t] = o
    # routing with scene splits
    scenes = sorted({r["scene"] for r in rows}); rng = np.random.default_rng(0)
    acc = collections.defaultdict(list)
    for _ in range(100):
        p = rng.permutation(len(scenes)); dev = {scenes[i] for i in p[: len(p) // 2]}
        per = collections.defaultdict(dict)
        for t in tasks:
            D = [r for r in rows if r["t"] == t and r["scene"] in dev]; T = [r for r in rows if r["t"] == t and r["scene"] not in dev]
            if not D or not T:
                continue
            cand = sorted({r["gu"] for r in D if np.isfinite(r["gu"])}) or [0.0]
            grid = [-np.inf] + list(np.quantile(cand, np.linspace(0, 1, 21))) + [np.inf]
            use = lambda r, tau: r["has_g"] and r["gu"] <= tau
            best = max(grid, key=lambda tau: np.mean([r["gs"] if use(r, tau) else r["ms"] for r in D]))
            route = np.array([use(r, best) for r in T]); ms = np.array([r["ms"] for r in T]); gs = np.array([r["gs"] for r in T])
            better_geo = np.mean([r["gs"] if r["has_g"] else r["ms"] for r in D]) > np.mean([r["ms"] for r in D])
            # two signals: also take the measurement when the MLLM itself is unsure
            cq = list(np.quantile([r["mc"] for r in D], np.linspace(0, 1, 11)))
            use2 = lambda r, a, b: r["has_g"] and (r["gu"] <= a or r["mc"] <= b)
            b1, b2 = max(((x, y) for x in grid for y in [-np.inf] + cq),
                         key=lambda xy: np.mean([r["gs"] if use2(r, *xy) else r["ms"] for r in D]))
            route2 = np.array([use2(r, b1, b2) for r in T])
            rnd = np.random.default_rng(len(acc[("mllm", t)])).random(len(T)) < route.mean()
            hasg = np.array([r["has_g"] for r in T])
            per[t] = dict(mllm=ms.mean(), geo=np.where(hasg, gs, ms).mean(), task=(np.where(hasg, gs, ms) if better_geo else ms).mean(),
                          hybrid=np.where(route, gs, ms).mean(), hybrid2=np.where(route2, gs, ms).mean(), random=np.where(rnd & hasg, gs, ms).mean(),
                          oracle=np.maximum(ms, gs).mean(), rate=route.mean())
        for t, d in per.items():
            for k, v in d.items():
                acc[(k, t)].append(v)
        if len(per) == len(tasks):
            for k in ("mllm", "geo", "task", "hybrid", "hybrid2", "random", "oracle"):
                acc[(k, "overall")].append(np.mean([per[t][k] for t in tasks]))
            acc[("gain", "overall")].append(np.mean([per[t]["hybrid"] - per[t]["mllm"] for t in tasks]))
            acc[("gain2", "overall")].append(np.mean([per[t]["hybrid2"] - per[t]["mllm"] for t in tasks]))
            acc[("gain2task", "overall")].append(np.mean([per[t]["hybrid2"] - per[t]["task"] for t in tasks]))
    res["routing"] = {f"{k}|{t}": dict(mean=float(np.mean(v)), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
                      for (k, t), v in acc.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'task':28s} {'n':>5s} {'MLLM':>6s} {'geo':>6s} {'orac':>6s} | {'mAURC':>6s} {'rand':>6s} {'rho':>6s} {'conf':>5s} | "
          f"{'nM':>4s} {'M-AURC':>6s} {'M-rnd':>6s} {'G-AURC':>6s} {'G-rnd':>6s}")
    for t, o in res["tasks"].items():
        print(f"{t:28s} {o['n']:5d} {100 * o['mllm']:6.1f} {100 * o['geo']:6.1f} {100 * o['oracle']:6.1f} | {o['mllm_aurc']:6.3f} "
              f"{o['mllm_aurc_random']:6.3f} {o['mllm_rho']:6.2f} {100 * o.get('mllm_mean_conf', np.nan):5.1f} | {o.get('matched_n', 0):4d} "
              f"{o.get('matched_mllm_aurc', np.nan):6.3f} {o.get('matched_mllm_random', np.nan):6.3f} {o.get('matched_geo_aurc', np.nan):6.3f} {o.get('matched_geo_random', np.nan):6.3f}")
    print("\nrouting (test scenes, mean over 100 splits)")
    for t in tasks + ["overall"]:
        line = {k: round(100 * res["routing"][f"{k}|{t}"]["mean"], 1) for k in ("mllm", "geo", "task", "random", "hybrid", "hybrid2", "oracle") if f"{k}|{t}" in res["routing"]}
        rate = res["routing"].get(f"rate|{t}", {}).get("mean")
        print(f"  {t:28s} {line} {'rate %.2f' % rate if rate is not None else ''}")
    for k in ("gain", "gain2", "gain2task"):
        g = res["routing"].get(f"{k}|overall")
        if g:
            print(f"  {k}: {100 * g['mean']:.1f} [{100 * g['p05']:.1f}, {100 * g['p95']:.1f}]")


if __name__ == "__main__":
    main()
