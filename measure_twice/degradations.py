"""Physically motivated, temporally consistent video degradations (5 severities each).

lowlight : sRGB->linear, exposure scaled by k, Poisson shot noise (photon budget) + Gaussian read noise,
           digital gain back to the original mean exposure, ->sRGB (camera auto-exposure behaviour).
motion   : blur along per-pixel optical flow (OpenCV DIS) to the next frame; exposure = fraction of the
           inter-frame motion, so moving objects blur more than static background.
codec    : real H.264 (libx264) encode/decode of the whole clip at a fixed QP.
haze     : atmospheric scattering I = J t + A (1 - t), t = exp(-beta * d) with GT depth d (Koschmieder).
"""
import os, shutil, subprocess, tempfile
import numpy as np
import cv2

SEVERITY = {
    "lowlight": [(1 / 8, 4000), (1 / 16, 4000), (1 / 32, 4000), (1 / 64, 4000), (1 / 128, 4000)],  # (exposure, full-well photons)
    "motion": [0.25, 0.5, 1.0, 1.5, 2.0],
    "codec": [22, 27, 32, 37, 42],
    "haze": [0.02, 0.04, 0.08, 0.12, 0.18],  # beta in 1/m (Sintel/Bonn depths in metres-ish)
}
# Held-out degradations: never used to train the adapter (generalization test, R008)
UNSEEN = {
    "defocus": [1.5, 3.0, 4.5, 6.0, 8.0],          # disk PSF radius in px
    "jpeg": [50, 30, 20, 12, 8],                    # per-frame JPEG quality (temporally inconsistent blocking)
    "overexpose": [1.5, 2.0, 3.0, 4.0, 6.0],        # linear exposure gain with sensor clipping
    "noise": [5, 10, 15, 25, 35],                   # additive Gaussian sensor noise, sigma in 8-bit units
    "night_motion": [0, 1, 2, 3, 4],                # compound: motion blur then low light at the same level
}


def _srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def lowlight(frames, level, seed=0):
    k, photons = SEVERITY["lowlight"][level]
    rng = np.random.default_rng(seed)
    out = []
    for f in frames:
        lin = _srgb2lin(f.astype(np.float32) / 255.0) * k
        shot = rng.poisson(lin * photons) / photons
        read = rng.normal(0, 2.0 / photons, lin.shape)
        noisy = np.clip(shot + read, 0, None) / k  # auto-exposure gain
        out.append((np.clip(_lin2srgb(noisy), 0, 1) * 255).astype(np.uint8))
    return out


def motion(frames, level):
    s = SEVERITY["motion"][level]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = []
    n = len(frames)
    for i, f in enumerate(frames):
        j = i + 1 if i + 1 < n else i - 1
        g0 = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[j], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(g0, g1, None) * (1 if j > i else -1)
        H, W = g0.shape
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        K = 9
        acc = np.zeros_like(f, dtype=np.float32)
        for t in np.linspace(-0.5, 0.5, K):
            mx = (xx + flow[..., 0] * np.float32(s * t)).astype(np.float32)
            my = (yy + flow[..., 1] * np.float32(s * t)).astype(np.float32)
            acc += cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        out.append(np.clip(acc / K, 0, 255).astype(np.uint8))
    return out


def codec(frames, level, tmp):
    qp = SEVERITY["codec"][level]
    os.makedirs(tmp, exist_ok=True)
    H, W = frames[0].shape[:2]
    H2, W2 = H - H % 2, W - W % 2
    raw = np.stack([f[:H2, :W2] for f in frames]).tobytes()
    vid = os.path.join(tmp, "c.mp4")
    enc = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}", "-r", "24",
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-qp", str(qp), "-pix_fmt", "yuv420p", vid]
    subprocess.run(enc, input=raw, check=True)
    dec = ["ffmpeg", "-loglevel", "error", "-i", vid, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(dec, capture_output=True, check=True).stdout
    arr = np.frombuffer(buf, np.uint8).reshape(-1, H2, W2, 3)[: len(frames)]
    out = []
    for f, a in zip(frames, arr):
        g = f.copy(); g[:H2, :W2] = a
        out.append(g)
    return out


def haze(frames, level, depths, airlight=0.85):
    beta = SEVERITY["haze"][level]
    out = []
    for f, d in zip(frames, depths):
        d = np.where(np.isfinite(d) & (d > 0), d, d[np.isfinite(d) & (d > 0)].max() if (d > 0).any() else 50.0)
        d = cv2.resize(d.astype(np.float32), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        t = np.exp(-beta * np.clip(d, 0, 200))[..., None]
        J = f.astype(np.float32) / 255.0
        out.append((np.clip(J * t + airlight * (1 - t), 0, 1) * 255).astype(np.uint8))
    return out


def defocus(frames, level):
    r = UNSEEN["defocus"][level]
    k = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:k, :k] - k // 2
    psf = ((xx ** 2 + yy ** 2) <= r ** 2).astype(np.float32); psf /= psf.sum()
    return [cv2.filter2D(f, -1, psf, borderType=cv2.BORDER_REFLECT) for f in frames]


def jpeg(frames, level):
    q = UNSEEN["jpeg"][level]
    out = []
    for f in frames:
        ok, b = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(b, cv2.IMREAD_COLOR)[..., ::-1].copy())
    return out


def overexpose(frames, level):
    g = UNSEEN["overexpose"][level]
    return [(_lin2srgb(_srgb2lin(f.astype(np.float32) / 255.0) * g) * 255).astype(np.uint8) for f in frames]


def noise(frames, level, seed=1):
    s = UNSEEN["noise"][level]; rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float32) + rng.normal(0, s, f.shape), 0, 255).astype(np.uint8) for f in frames]


def apply(kind, frames, level, depths=None, tmp=os.path.join(tempfile.gettempdir(), "mt_codec")):
    if kind == "defocus":
        return defocus(frames, level)
    if kind == "jpeg":
        return jpeg(frames, level)
    if kind == "overexpose":
        return overexpose(frames, level)
    if kind == "noise":
        return noise(frames, level)
    if kind == "night_motion":
        return lowlight(motion(frames, level), level)
    if kind == "clean":
        return frames
    if kind == "lowlight":
        return lowlight(frames, level)
    if kind == "motion":
        return motion(frames, level)
    if kind == "codec":
        return codec(frames, level, tmp)
    if kind == "haze":
        return haze(frames, level, depths)
    raise ValueError(kind)
