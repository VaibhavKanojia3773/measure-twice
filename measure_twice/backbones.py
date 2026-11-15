"""Installers and loaders for the three metric multi-view reconstructors used in the paper.

  mapanything  MapAnything (Keetha et al., 3DV 2026)            facebook/map-anything
  pi3x         pi3-X, metric-scale head (Wang et al., ICLR 2026)  yyfz233/Pi3X
  da3m         Depth Anything 3, nested giant+large metric model   depth-anything/DA3NESTED-GIANT-LARGE

Each repository is cloned into MT_WORK on first use. All inference runs in fp16: the T4 GPUs used for the paper
(compute capability 7.5) have no bf16 support.
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from .utils import sh, log, WORK

DTYPE = torch.float16
# Recent PyTorch reports bf16 as available on pre-Ampere GPUs (emulated). Models that choose bf16 from that flag then miss
# the memory-efficient attention kernel and fall back to one that materialises the full attention matrix. Force fp16.
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    torch.cuda.is_bf16_supported = lambda *a, **k: False


def load_frames(paths, width=518):
    """RGB frames resized to `width` and a height divisible by 14, as an (N, 3, h, w) tensor in [0, 1]."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    W, H = ims[0].size
    h = int(round(H * width / W / 14)) * 14
    return torch.stack([torch.from_numpy(np.asarray(im.resize((width, h), Image.BICUBIC))).permute(2, 0, 1)
                        for im in ims]).float() / 255.0


def install(name):
    """Clone and install a reconstructor once; later calls only extend sys.path."""
    marker = os.path.join(WORK, f".installed_{name}")
    if os.path.exists(marker):
        sys.path.insert(0, open(marker).read().strip())
        return
    if name == "mapanything":
        root = os.path.join(WORK, "map-anything")
        if not os.path.isdir(root):
            sh(f"git clone --depth 1 https://github.com/facebookresearch/map-anything.git {root}")
            rc, _ = sh(f"cd {root} && pip install -q -e .", check=False)
            if rc != 0:
                sh(f"cd {root} && pip install -q --no-deps -e . && pip install -q uniception einops huggingface_hub safetensors", check=False)
        path = root
    elif name == "pi3x":
        root = os.path.join(WORK, "Pi3")
        if not os.path.isdir(root):
            sh(f"git clone --depth 1 https://github.com/yyfz/Pi3.git {root}")
        # pi3-X rescales camera translations in place after using them; replace by an out-of-place equivalent.
        f = os.path.join(root, "pi3", "models", "pi3x.py"); src = open(f).read()
        bad = "camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)"
        good = ("camera_poses = torch.cat([torch.cat([camera_poses[..., :3, :3], (camera_poses[..., :3, 3] * metric.view(B, 1, 1))[..., None]], -1), "
                "camera_poses[..., 3:, :]], -2)")
        if bad in src:
            open(f, "w").write(src.replace(bad, good)); log("patched pi3x in-place pose scaling")
        sh("pip install -q huggingface_hub safetensors einops", check=False)
        path = root
    elif name == "da3m":
        root = os.path.join(WORK, "Depth-Anything-3")
        if not os.path.isdir(root):
            sh(f"git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git {root}")
        sh(f"cd {root} && pip install -q --no-deps --ignore-requires-python -e .", check=False)
        for pkg in ("addict", "einops", "omegaconf", "huggingface_hub", "safetensors", "imageio", "trimesh", "plyfile",
                    "opencv-python-headless", "evo", "moviepy", "pillow_heif", "pycolmap"):
            sh(f"pip install -q {pkg}", check=False, tail=500)
        path = os.path.join(root, "src")
    else:
        raise ValueError(f"unknown reconstructor {name!r}")
    sys.path.insert(0, path)
    open(marker, "w").write(path)


def load(name, device="cuda"):
    """Return the pretrained network for `name` on `device`."""
    install(name)
    if name == "mapanything":
        from mapanything.models import MapAnything
        return MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
    if name == "pi3x":
        from pi3.models.pi3x import Pi3X
        return Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    if name == "da3m":
        from depth_anything_3.api import DepthAnything3
        return DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device=device)
    raise ValueError(f"unknown reconstructor {name!r}")
