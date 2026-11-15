"""Metric multi-view reconstruction with per-pixel confidence.

Recon(name).run(paths) -> dict(
    pts   (N, h, w, 3) world points in metres,
    conf  (N, h, w)    the model's own per-pixel confidence (the "native 3D confidence" baseline of the paper),
    c2w   (N, 4, 4)    camera-to-world poses)
Inputs are frames already resized to width 518 with a height divisible by 14, so no model crops them.
Every run returns its own coordinate frame; runs are never registered to each other.
"""
import numpy as np
import torch
from .utils import log
from .backbones import load, load_frames, DTYPE


class Recon:
    def __init__(self, name, device="cuda"):
        self.name, self.dev = name, device
        self.m = load(name, device)
        self._logged = False

    @torch.no_grad()
    def run(self, paths):
        if self.name == "mapanything":
            from mapanything.utils.image import load_images
            preds = self.m.infer(load_images(list(paths)), memory_efficient_inference=True, use_amp=True, amp_dtype="fp16",
                                 apply_mask=False, mask_edges=False, apply_confidence_mask=False)
            if not self._logged:
                log("mapanything keys", sorted(preds[0].keys()), "pts3d", tuple(preds[0]["pts3d"].shape)); self._logged = True
            pts = np.stack([p["pts3d"][0].float().cpu().numpy() for p in preds])
            conf = np.stack([p["conf"][0].float().cpu().numpy() for p in preds])
            conf = conf[..., 0] if conf.ndim == 4 else conf
            c2w = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds]).astype(np.float64)
            return dict(pts=pts.astype(np.float32), conf=conf.astype(np.float32), c2w=c2w)
        if self.name == "da3m":
            p = self.m.inference(list(paths))
            if not self._logged:
                log("da3m depth", p.depth.shape, "is_metric", p.is_metric, "scale", p.scale_factor); self._logged = True
            D = np.asarray(p.depth, np.float32); K = np.asarray(p.intrinsics, np.float64); E = np.asarray(p.extrinsics, np.float64)
            n, h, w = D.shape
            u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
            pts, c2w = [], []
            for i in range(n):                          # back-project metric depth with the predicted cameras
                M = np.eye(4); M[:E.shape[1], :4] = E[i][:E.shape[1]]; C = np.linalg.inv(M)
                X = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * D[i], (v - K[i, 1, 2]) / K[i, 1, 1] * D[i], D[i]], -1)
                pts.append(X @ C[:3, :3].T + C[:3, 3]); c2w.append(C)
            conf = np.asarray(p.conf, np.float32) if p.conf is not None else np.ones_like(D)
            return dict(pts=np.stack(pts).astype(np.float32), conf=conf, c2w=np.stack(c2w))
        x = load_frames(paths).to(self.dev)              # pi3x
        with torch.amp.autocast("cuda", dtype=DTYPE):
            r = self.m(x[None])
        if not self._logged:
            log("pi3x keys", sorted(r.keys()), "points", tuple(r["points"].shape), "metric", float(r["metric"].float().mean()))
            self._logged = True
        pts = r["points"][0].float().cpu().numpy()
        conf = torch.sigmoid(r["conf"][0, ..., 0].float()).cpu().numpy()    # logits -> [0, 1]
        c2w = r["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        return dict(pts=pts.astype(np.float32), conf=conf.astype(np.float32), c2w=c2w)
