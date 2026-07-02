"""Shared utilities for the FL gradient-reconstruction pipeline.

No torchvision / matplotlib dependency (not in the JURECA venv). Grids are
rendered with PIL. All paths are configurable via env vars so the same code
runs on JURECA and locally.
"""
import os
import json
import math
import numpy as np
import torch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "FL_DATA_ROOT",
    "/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction",
)
WORK_ROOT = os.environ.get(
    "FL_WORK_ROOT",
    "/p/scratch/training2625/zainzinger1/1337/recon_work",
)
MODELS = [f"model{i}" for i in range(1, 13)]

IMG_C, IMG_H, IMG_W = 3, 64, 64
IMG_NUMEL = IMG_C * IMG_H * IMG_W  # 12288
N_IMAGES = 128


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def work_path(*parts):
    return os.path.join(WORK_ROOT, *parts)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_grad(name, map_location="cpu"):
    """Return the gradient bundle dict for a model name (e.g. 'model1')."""
    return torch.load(
        os.path.join(DATA_ROOT, "gradients", f"{name}.pt"),
        weights_only=False, map_location=map_location,
    )


def load_state(name, map_location="cpu"):
    """Return the model state_dict for a model name."""
    return torch.load(
        os.path.join(DATA_ROOT, "models", f"{name}.pt"),
        weights_only=False, map_location=map_location,
    )


# ----------------------------------------------------------------------------
# Image ops
# ----------------------------------------------------------------------------
def total_variation(x):
    """Anisotropic TV. x: (...,C,H,W). Returns mean per-image TV."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-3, -2, -1))
    return dh + dw


def per_image_minmax(x, eps=1e-6):
    """Min-max normalise each image in a batch to [0,1]. x: (N,C,H,W)."""
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + eps)
    return flat.reshape_as(x)


def image_stats(x):
    """Per-image diagnostic stats. x: (N,C,H,W) raw (unclamped).
    Returns dict of 1-D tensors length N."""
    N = x.shape[0]
    flat = x.reshape(N, -1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    rng_frac = ((safe >= -0.05) & (safe <= 1.05)).float().mean(dim=1)
    std = safe.std(dim=1)
    mn = safe.min(dim=1).values
    mx = safe.max(dim=1).values
    xc = safe.clamp(0, 1).reshape(N, IMG_C, IMG_H, IMG_W)
    tv = total_variation(xc)
    return {
        "finite": finite, "rng_frac": rng_frac, "std": std,
        "min": mn, "max": mx, "tv": tv,
    }


# ----------------------------------------------------------------------------
# Grid rendering (PIL)
# ----------------------------------------------------------------------------
def save_image_grid(x, path, ncol=16, pad=1, upscale=2):
    """Save a batch of images (N,3,64,64) in [0,1] as a PNG grid via PIL."""
    from PIL import Image
    x = x.detach().float().clamp(0, 1).cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    ncol = min(ncol, max(1, N))
    nrow = math.ceil(N / ncol)
    grid = np.ones((nrow * (H + pad) + pad, ncol * (W + pad) + pad, 3), dtype=np.float32)
    for idx in range(N):
        r, c = divmod(idx, ncol)
        img = x[idx].permute(1, 2, 0).numpy()
        y0 = r * (H + pad) + pad
        x0 = c * (W + pad) + pad
        grid[y0:y0 + H, x0:x0 + W, :] = img
    im = Image.fromarray((grid * 255).astype(np.uint8))
    if upscale != 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.NEAREST)
    ensure_dir(os.path.dirname(path))
    im.save(path)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------
# Gradient flattening / matching helpers (used by the optimisation stage)
# ----------------------------------------------------------------------------
def flatten_grads(grads_dict, selected_keys=None):
    keys = list(grads_dict.keys()) if selected_keys is None else selected_keys
    vecs = [grads_dict[k].reshape(-1) for k in keys if grads_dict.get(k) is not None]
    return torch.cat(vecs) if vecs else torch.zeros(0)
