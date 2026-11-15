"""Geometry-grounded VSI-Bench answers with error bars, from the per-scene 3D memory written by vsi_worker (runs on CPU).

For every reconstruction run (model x frame subset) a small spatial program answers each question from the 3D points:
  object_counting        number of 3D instances (detections clustered by their 3D centres)
  object_abs_distance    closest-point distance between the two objects (m)
  object_size_estimation longest side of the gravity-aligned box of the object (cm)
  room_size_estimation   floor area covered by the reconstruction (m^2)
  object_rel_distance    option with the smallest closest-point distance to the target
  object_rel_direction_* side of the queried object, standing at A and facing B
  obj_appearance_order   order of first detection in the 32 frames
The runs form an ensemble: the point answer is the ensemble median (numbers) or vote (choices), and the spread gives the
uncertainty that split-conformal calibration turns into intervals with guaranteed coverage.

python -m measure_twice.programs --npz outputs/scene_memory --questions measure_twice/data/vsi_test.txt --out results/vsi_answers.json
"""
import argparse, glob, json, os, re
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

NUM = ("object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation")


# ------------------------------------------------------------------ scene geometry per run
def up_axis(c2w, pts):
    """Gravity 'up' from camera orientation (cameras are held upright: image y points down), refined by the floor plane."""
    up = -np.mean(c2w[:, :3, 1], 0); up /= np.linalg.norm(up)
    h = pts @ up
    low = pts[h < np.percentile(h, 15)]
    if len(low) > 200:                                  # floor plane by PCA of the lowest points, if roughly horizontal
        c = low - low.mean(0)
        n = np.linalg.svd(c[np.random.default_rng(0).choice(len(c), min(4000, len(c)), replace=False)], full_matrices=False)[2][-1]
        n = n if n @ up > 0 else -n
        if n @ up > np.cos(np.radians(20)):
            up = n
    return up


def basis(up):
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def trim(P, C=None, q=0.0):
    """Drop low-confidence points and outliers (beyond 2.5 MAD of the median distance to the median)."""
    if C is not None and q > 0 and len(P) > 20:
        P = P[C >= np.quantile(C, q)]
    if len(P) < 5:
        return P
    d = np.linalg.norm(P - np.median(P, 0), axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    return P[d < np.median(d) + 2.5 * 1.4826 * mad]


class RunScene:
    def __init__(self, Z, run, cats, min_score=0.35):
        self.ok = f"{run}__spts" in Z
        if not self.ok:
            return
        self.c2w = Z[f"{run}__c2w"].astype(np.float64)
        sp = Z[f"{run}__spts"].astype(np.float32); sc = Z[f"{run}__sconf"].astype(np.float32)
        good = np.isfinite(sp).all(1)
        self.spts, self.sconf = sp[good], sc[good]
        self.up = up_axis(self.c2w, self.spts)
        self.e1, self.e2 = basis(self.up)
        self.frames = Z[f"{run}__frames"]
        # detections -> instances per category
        self.inst = {}
        if f"{run}__dpts" not in Z:
            return
        dp = Z[f"{run}__dpts"].astype(np.float32); dc = Z[f"{run}__dconf"].astype(np.float32); di = Z[f"{run}__did"]
        score, cat = Z["det_score"], Z["det_cat"]
        per, self.cconf, self.cdet = {}, {}, {}
        for k in np.unique(di):
            if score[k] < min_score:
                continue
            P = trim(dp[di == k], dc[di == k], 0.2)
            P = P[np.isfinite(P).all(1)]
            if len(P) >= 10:
                c = cats[cat[k]]
                per.setdefault(c, []).append((P, float(score[k])))
                self.cconf.setdefault(c, []).append(float(np.median(dc[di == k])))      # native per-point confidence
                self.cdet[c] = max(self.cdet.get(c, 0.0), float(score[k]))             # detector confidence
        for c, dets in per.items():
            self.inst[c] = cluster(dets)

    def obj(self, c):
        """The largest instance of category c (most supporting detections), or None."""
        L = self.inst.get(c)
        return None if not L else max(L, key=lambda x: (x[1], len(x[0])))[0]


def cluster(dets, link=0.6):
    """Single-linkage clustering of detections whose 3D extents overlap or whose centres are within `link` metres."""
    n = len(dets); parent = list(range(n))
    cen = [np.median(P, 0) for P, _ in dets]
    rad = [np.percentile(np.linalg.norm(P - c, axis=1), 80) for (P, _), c in zip(dets, cen)]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(cen[i] - cen[j]) < max(link, 0.7 * (rad[i] + rad[j])):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [(trim(np.concatenate([dets[i][0] for i in g])), len(g)) for g in groups.values()]


def closest(A, B):
    """Robust closest-point distance: 2nd percentile of nearest-neighbour distances from the smaller set to the larger."""
    if len(A) > len(B):
        A, B = B, A
    d = cKDTree(B).query(A)[0]
    return float(np.percentile(d, 2))


def longest_side(rs, P):
    h = P @ rs.up; x = P @ rs.e1; y = P @ rs.e2
    best = None
    for ang in np.radians(np.arange(0, 90, 5)):        # minimum-area horizontal rectangle
        u = x * np.cos(ang) + y * np.sin(ang); v = -x * np.sin(ang) + y * np.cos(ang)
        a, b = np.percentile(u, 98) - np.percentile(u, 2), np.percentile(v, 98) - np.percentile(v, 2)
        if best is None or a * b < best[0]:
            best = (a * b, a, b)
    return float(max(best[1], best[2], np.percentile(h, 98) - np.percentile(h, 2)))


def floor_area(rs, cell=0.1, conf_q=0.3):
    P = rs.spts[rs.sconf >= np.quantile(rs.sconf, conf_q)]
    x, y = P @ rs.e1, P @ rs.e2
    lo = np.percentile(np.stack([x, y], 1), 0.5, 0); hi = np.percentile(np.stack([x, y], 1), 99.5, 0)
    keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
    ix = ((x[keep] - lo[0]) / cell).astype(int); iy = ((y[keep] - lo[1]) / cell).astype(int)
    G = np.zeros((ix.max() + 1, iy.max() + 1), bool); G[ix, iy] = True
    G = ndimage.binary_closing(G, iterations=3)
    G = ndimage.binary_fill_holes(G)
    return float(G.sum() * cell * cell)


def answer(rs, q, cats_in_q):
    t = q["question_type"]
    if t == "room_size_estimation":
        return floor_area(rs)
    if t == "object_counting":
        L = rs.inst.get(cats_in_q[0]); return float(len(L)) if L else 0.0
    if t == "object_abs_distance":
        A, B = rs.obj(cats_in_q[0]), rs.obj(cats_in_q[1])
        return None if A is None or B is None else closest(A, B)
    if t == "object_size_estimation":
        A = rs.obj(cats_in_q[0]); return None if A is None else 100.0 * longest_side(rs, A)
    if t == "object_rel_distance":
        *opts, tgt = cats_in_q
        T = rs.obj(tgt)
        if T is None:
            return None
        d = [closest(rs.obj(o), T) if rs.obj(o) is not None else np.inf for o in opts]
        return None if not np.isfinite(min(d)) else "ABCD"[int(np.argmin(d))]
    if t.startswith("object_rel_direction"):
        A, B, C = (rs.obj(c) for c in cats_in_q)
        if A is None or B is None or C is None:
            return None
        flat = lambda v: v - (v @ rs.up) * rs.up
        pa, pb, pc = (np.median(P, 0) for P in (A, B, C))
        f = flat(pb - pa); f /= np.linalg.norm(f) + 1e-9
        r = flat(pc - pa); r /= np.linalg.norm(r) + 1e-9
        side = np.cross(rs.up, f) @ r                     # > 0: left of the viewer standing at A facing B
        lr = "left" if side > 0 else "right"
        fb = "front" if r @ f > 0 else "back"
        opts = [o.split(". ", 1)[1].lower() for o in q["options"]]
        if any("front-left" in o for o in opts):
            want = f"{fb}-{lr}"
        elif "back" in opts and np.degrees(np.arccos(np.clip(r @ f, -1, 1))) >= 135:
            want = "back"
        else:
            want = lr
        return "ABCD"[opts.index(want)] if want in opts else None
    if t == "obj_appearance_order":
        first = {}
        for c in cats_in_q:
            fr = [f for f, k in zip(rs.det_frame, rs.det_catname) if k == c]
            first[c] = min(fr) if fr else 99
        order = sorted(cats_in_q, key=lambda c: first[c])
        opts = [[x.strip() for x in o.split(". ", 1)[1].split(",")] for o in q["options"]]
        return "ABCD"[opts.index(order)] if order in opts else None
    return None


def native(rs, q, cats_in_q):
    """The reconstructor's own confidence for the evidence a question uses (median per-point confidence of its objects,
    or of the scene points for room size) and the detector score; baselines for the ensemble spread."""
    if q["question_type"] == "room_size_estimation":
        return float(np.median(rs.sconf)), None
    c = [x for x in cats_in_q if x in getattr(rs, "cconf", {})]
    if not c:
        return None, None
    return float(np.mean([np.median(rs.cconf[x]) for x in c])), float(np.mean([rs.cdet[x] for x in c]))


def q_cats(q):
    s, t = q["question"], q["question_type"]
    pats = {"object_counting": r"How many (.+?)\(s\) are in this room",
            "object_abs_distance": r"distance between the (.+?) and the (.+?) \(in meters\)",
            "object_size_estimation": r"dimension \(length, width, or height\) of the (.+?), measured",
            "object_rel_distance": r"which of these objects \((.+?)\) is the closest to the (.+?)\?",
            "obj_appearance_order": r"following categories in the video: (.+?)\?"}
    if t in pats:
        m = re.search(pats[t], s, re.S)
        return [x.strip().lower() for g in m.groups() for x in g.split(",")] if m else None
    if t.startswith("object_rel_direction"):
        m = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) to ", s, re.S)
        return [x.lower() for x in m.groups()] if m else None
    return [] if t == "room_size_estimation" else None


def mra(pred, gt):
    rel = abs(pred - gt) / max(abs(gt), 1e-9)
    return float(np.mean([rel < 1 - th for th in np.arange(0.5, 1.0, 0.05)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+"); ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {}                                   # a scene may be split over several extraction jobs (one per model family)
    for d in a.npz:
        for f in glob.glob(f"{d}/*.npz"):
            files.setdefault(os.path.basename(f)[:-4], []).append(f)
    Q = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
    rows = []
    for key, fs in sorted(files.items()):
        RS = {}
        for f in fs:                             # each file carries its own detections; runs index into those
            Z = dict(np.load(f, allow_pickle=True))
            cats = [str(c) for c in Z["cats"]]
            for r in sorted({k.split("__")[0] for k in Z if k.endswith("__spts")}):
                rs = RunScene(Z, r, cats)
                if rs.ok and r not in RS:
                    rs.det_frame = list(Z["det_frame"]); rs.det_catname = [cats[c] for c in Z["det_cat"]]
                    RS[r] = rs
        ds, sc, *cond = key.split("__")              # degraded copies carry a condition suffix
        cond = cond[0] if cond else "clean"
        for q in Q:
            if q["dataset"] != ds or q["scene_name"] != sc or q["question_type"] == "route_planning":
                continue
            cq = q_cats(q)
            if cq is None:
                continue
            per, nat, det = {}, {}, {}
            for r, rs in RS.items():
                try:
                    v = answer(rs, q, cq)
                except Exception:
                    v = None
                per[r] = v
                if v is not None:
                    nat[r], det[r] = native(rs, q, cq)
            rows.append(dict(id=q["id"], type=q["question_type"], ds=ds, scene=sc, cond=cond, gt=q["ground_truth"], runs=per,
                             native=nat, det=det))
    json.dump(rows, open(a.out, "w"), indent=0)
    print(len(rows), "answered rows from", len(files), "scenes ->", a.out)


if __name__ == "__main__":
    main()
