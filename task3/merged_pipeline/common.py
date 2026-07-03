"""Shared utilities for the MERGED pipeline (teammate's 0.3888 approach + our levers).

Self-contained: no dependency on the teammate's JURECA scratch modules. Points at
the LOCAL task3/gradients + task3/models by default. Primitives (image_stats, TV,
gaussian_blur, quality_score, per_image_minmax) are ported verbatim from the
teammate's utils.py/postprocess.py so behaviour matches the 0.3888 pipeline.
"""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths — default to the LOCAL task3 folder (parent of this dir)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TASK3 = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("FL_DATA_ROOT", TASK3)          # expects gradients/ + models/
WORK_ROOT = os.environ.get("FL_WORK_ROOT", os.path.join(HERE, "work"))
MODELS = [f"model{i}" for i in range(1, 13)]

IMG_C, IMG_H, IMG_W = 3, 64, 64
IMG_NUMEL = IMG_C * IMG_H * IMG_W                           # 12288
N_IMAGES = 128


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def work_path(*parts):
    return os.path.join(WORK_ROOT, *parts)


def load_grad(name, map_location="cpu"):
    return torch.load(os.path.join(DATA_ROOT, "gradients", f"{name}.pt"),
                      weights_only=False, map_location=map_location)


def load_state(name, map_location="cpu"):
    return torch.load(os.path.join(DATA_ROOT, "models", f"{name}.pt"),
                      weights_only=False, map_location=map_location)


# ---------------------------------------------------------------------------
# Image ops (ported verbatim from teammate utils.py / postprocess.py)
# ---------------------------------------------------------------------------
def total_variation(x):
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-3, -2, -1))
    return dh + dw


def per_image_minmax(x, eps=1e-6):
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo + eps)).reshape_as(x)


def image_stats(x):
    N = x.shape[0]
    flat = x.reshape(N, -1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    rng_frac = ((safe >= -0.05) & (safe <= 1.05)).float().mean(dim=1)
    std = safe.std(dim=1)
    xc = safe.clamp(0, 1).reshape(N, IMG_C, IMG_H, IMG_W)
    tv = total_variation(xc)
    return {"finite": finite, "rng_frac": rng_frac, "std": std,
            "min": safe.min(dim=1).values, "max": safe.max(dim=1).values, "tv": tv}


def quality_score(imgs):
    """0.3888 quality signal: rewards in-range + moderate contrast, penalises grain."""
    st = image_stats(imgs)
    contrast = torch.clamp(st["std"] / 0.25, max=1.0)
    q = st["rng_frac"] + 0.3 * contrast - 2.0 * st["tv"]
    return torch.where(st["finite"], q, torch.full_like(q, -1e9))


def _gaussian_kernel(sigma, k):
    ax = torch.arange(k) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(imgs, sigma=1.0, k=5):
    """Separable depthwise Gaussian blur (teammate's, no torchvision)."""
    g = _gaussian_kernel(sigma, k).to(imgs.dtype)
    kx = g.view(1, 1, 1, k).repeat(3, 1, 1, 1)
    ky = g.view(1, 1, k, 1).repeat(3, 1, 1, 1)
    p = k // 2
    x = F.pad(imgs, (p, p, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=3)
    x = F.pad(x, (0, 0, p, p), mode="reflect")
    x = F.conv2d(x, ky, groups=3)
    return x


def feat_vectors(imgs, size=16):
    """Downsampled gray feature for dedup / farthest-point diversity (v3)."""
    g = imgs.mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1)


def save_image_grid(x, path, ncol=16, pad=1, upscale=2):
    from PIL import Image
    x = x.detach().float().clamp(0, 1).cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    ncol = min(ncol, max(1, N))
    nrow = math.ceil(N / ncol)
    grid = np.ones((nrow * (H + pad) + pad, ncol * (W + pad) + pad, 3), np.float32)
    for idx in range(N):
        r, c = divmod(idx, ncol)
        grid[r * (H + pad) + pad:r * (H + pad) + pad + H,
             c * (W + pad) + pad:c * (W + pad) + pad + W] = x[idx].permute(1, 2, 0).numpy()
    im = Image.fromarray((grid * 255).astype(np.uint8))
    if upscale != 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.NEAREST)
    ensure_dir(os.path.dirname(path))
    im.save(path)
    return path
