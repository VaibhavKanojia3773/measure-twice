"""VSI-Bench: per-scene probabilistic 3D memory for geometry-grounded answers with error bars.

For each scene video: 32 uniformly sampled frames; Grounding DINO + SAM masks for every object category its questions
mention; metric reconstructions from several models and frame subsets (the ensemble that gives the uncertainty).
Saved per scene (OUT/vsi/<dataset>__<scene>.npz):
  frames            sampled frame indices in the video
  det_*             detections: frame, category id, score, box, mask area
  <run>__c2w        camera-to-world poses of the frames in the run (run = <model>_<subset>)
  <run>__frames     which of the 32 frames the run used
  <run>__dpts/dconf/did   up to 400 world points per detection (with confidence and detection id)
  <run>__spts/sconf/sfr   40k random scene points (room size, floor plane)
Stages run model by model so that only one large network is on the GPU at a time.

python -m measure_twice.scene_memory --scenes arkitscenes/41069025 ... --root data/vsi --budget_h 4
"""
import argparse, glob, json, os, time
import numpy as np
from PIL import Image
from .utils import log, WORK, OUT

N_FRAMES, SUB, N_SUB, W = 32, 16, 2, 518
CATS = json.load(open(os.path.join(os.path.dirname(__file__), "data", "vsi_cats.json")))


def frames_of(video, n=N_FRAMES):
    import cv2
    cap = cv2.VideoCapture(video)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, tot - 1, n).round().astype(int)
    ims = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        ims.append((int(i), Image.fromarray(f[:, :, ::-1])))
    return ims


def resize(im):
    w, h = im.size
    nh = int(round(h * W / w / 14)) * 14
    return im.resize((W, nh), Image.BICUBIC)


def runs_for(model, n):
    rng = np.random.default_rng(0)
    rs = {f"{model}_all": np.arange(n)}
    for k in range(N_SUB):
        rs[f"{model}_s{k}"] = np.sort(rng.choice(n, size=min(SUB, n), replace=False))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+"); ap.add_argument("--root", required=True)
    ap.add_argument("--models", default="mapanything,pi3x"); ap.add_argument("--budget_h", type=float, default=4.0)
    ap.add_argument("--only", default="", help="json {model: [scenes]} restricting a model to some scenes")
    ap.add_argument("--degrade", default="", help="kind:level from measure_twice.degradations applied to the sampled frames, e.g. lowlight:2")
    a = ap.parse_args()
    t0 = time.time()
    over = lambda: time.time() - t0 > a.budget_h * 3600
    tmp = f"{WORK}/vsi_tmp"; os.makedirs(tmp, exist_ok=True); os.makedirs(f"{OUT}/vsi", exist_ok=True)
    suffix = f"__{a.degrade.replace(':', '')}" if a.degrade else ""
    keyof = lambda s: s.replace("/", "__") + suffix
    todo = [s for s in a.scenes if not os.path.exists(f"{OUT}/vsi/{keyof(s)}.npz")]

    # stage 1: frames + detections
    from .grounding import Grounder
    G = Grounder()
    for s in todo:
        key = keyof(s); fd = f"{tmp}/{key}"
        if os.path.exists(f"{fd}/dets.npz"):
            continue
        if over():
            log("budget reached in detection"); break
        ds, sc = s.split("/")
        vids = glob.glob(f"{a.root}/**/{sc}.mp4", recursive=True)
        if not vids:
            log("no video", s); continue
        os.makedirs(fd, exist_ok=True)
        ims = frames_of(vids[0])
        if a.degrade:                                   # physically based degradation of the sampled frames (shared by all models)
            from . import degradations as DG
            kind, lvl = a.degrade.split(":")
            arr = DG.apply(kind, [np.asarray(resize(im)) for _, im in ims], int(lvl), tmp=f"{tmp}/codec_{key}")
            ims = [(fi, Image.fromarray(x)) for (fi, _), x in zip(ims, arr)]
        cats = CATS.get(s, [])
        det = dict(frame=[], cat=[], score=[], box=[], area=[]); masks = []
        for j, (fi, im) in enumerate(ims):
            r = resize(im); r.save(f"{fd}/{j:02d}.jpg", quality=95)
            os.makedirs(f"{OUT}/frames/{key}", exist_ok=True); r.save(f"{OUT}/frames/{key}/{j:02d}.jpg", quality=85)
            for c, sc_, b, m in (G.detect(r, cats) if cats else []):
                det["frame"].append(j); det["cat"].append(cats.index(c)); det["score"].append(sc_)
                det["box"].append(b); det["area"].append(int(m.sum())); masks.append(np.packbits(m))
        np.savez_compressed(f"{fd}/dets.npz", frames=np.array([fi for fi, _ in ims]), cats=np.array(cats, dtype=object),
                            shape=np.array([r.size[1], r.size[0]]),
                            masks=np.stack(masks).astype(np.uint8) if masks else np.zeros((0, 1), np.uint8),
                            **{f"det_{k}": np.array(v) for k, v in det.items()})
        log(f"det {s}: {len(ims)} frames, {len(det['frame'])} detections over {len(cats)} categories")
    del G
    import torch; torch.cuda.empty_cache()

    # stage 2: reconstructions, one model at a time
    from .recon import Recon
    import torch
    rng = np.random.default_rng(0)
    only = json.load(open(a.only)) if a.only else {}
    for model in a.models.split(","):
        try:
            R = Recon(model)
        except Exception as e:
            log("cannot load", model, repr(e)[:300]); continue
        for s in todo:
            key = keyof(s); fd = f"{tmp}/{key}"
            if not os.path.exists(f"{fd}/dets.npz") or os.path.exists(f"{fd}/{model}.npz"):
                continue
            if model in only and s not in only[model]:
                continue
            if over():
                log("budget reached in", model); break
            D = np.load(f"{fd}/dets.npz", allow_pickle=True)
            H, Wd = D["shape"]; paths = sorted(glob.glob(f"{fd}/[0-9][0-9].jpg"))
            res = {}
            try:
                for run, sel in runs_for(model, len(paths)).items():
                    tt = time.time()
                    try:
                        o = R.run([paths[i] for i in sel])
                    except torch.cuda.OutOfMemoryError:           # retry on a 24-frame subset (keeps the run usable)
                        torch.cuda.empty_cache()
                        sel = sel[np.linspace(0, len(sel) - 1, min(24, len(sel))).round().astype(int)]
                        o = R.run([paths[i] for i in sel])
                    torch.cuda.empty_cache()
                    h, w = o["pts"].shape[1:3]
                    res[f"{run}__c2w"] = o["c2w"].astype(np.float32); res[f"{run}__frames"] = sel
                    pos = {int(f): k for k, f in enumerate(sel)}
                    dp, dc, di = [], [], []
                    for k, f in enumerate(D["det_frame"]):
                        if int(f) not in pos:
                            continue
                        m = np.unpackbits(np.asarray(D["masks"][k], dtype=np.uint8))[:H * Wd].reshape(H, Wd).astype(bool)
                        if (h, w) != (H, Wd):
                            m = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
                        P = o["pts"][pos[int(f)]][m]; C = o["conf"][pos[int(f)]][m]
                        ok = np.isfinite(P).all(1)
                        P, C = P[ok], C[ok]
                        if len(P) > 400:
                            pick = rng.choice(len(P), 400, replace=False); P, C = P[pick], C[pick]
                        dp.append(P); dc.append(C); di.append(np.full(len(P), k))
                    if dp:
                        res[f"{run}__dpts"] = np.concatenate(dp).astype(np.float16)
                        res[f"{run}__dconf"] = np.concatenate(dc).astype(np.float16)
                        res[f"{run}__did"] = np.concatenate(di).astype(np.int32)
                    P = o["pts"].reshape(-1, 3); C = o["conf"].reshape(-1); F = np.repeat(sel, h * w)
                    ok = np.flatnonzero(np.isfinite(P).all(1))
                    pick = rng.choice(ok, min(40000, len(ok)), replace=False)
                    res[f"{run}__spts"] = P[pick].astype(np.float16); res[f"{run}__sconf"] = C[pick].astype(np.float16)
                    res[f"{run}__sfr"] = F[pick].astype(np.int16)
                    log(f"{model} {s} {run}: {len(sel)} frames {h}x{w} in {time.time() - tt:.1f}s")
            except Exception as e:
                log("FAILED", model, s, repr(e)[:300])
                import torch; torch.cuda.empty_cache()
                continue
            np.savez(f"{fd}/{model}.npz", **res)
        del R
        import torch; torch.cuda.empty_cache()

    # stage 3: merge
    for s in todo:
        key = keyof(s); fd = f"{tmp}/{key}"
        if not os.path.exists(f"{fd}/dets.npz"):
            continue
        D = dict(np.load(f"{fd}/dets.npz", allow_pickle=True))
        D.pop("masks", None)
        for f in glob.glob(f"{fd}/*.npz"):
            if not f.endswith("dets.npz"):
                D.update(dict(np.load(f)))
        np.savez_compressed(f"{OUT}/vsi/{key}.npz", **D)
    log("worker done", len(todo), "scenes", f"{(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
