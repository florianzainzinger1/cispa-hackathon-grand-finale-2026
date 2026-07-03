"""Stages 4-6: metric-level SSIM optimisation (the teammate's biggest levers) plus
OUR CNN chroma correction.

Order (matches the 0.3888 climb):
  0. OUR chroma fix on CNN models (model10 desaturate, model12 color-cast)   <-- OURS
  4. global metric denoise: Gaussian blur sigma=1.0, then NLM h=0.05          (+0.031)
  5. weak/dead-model mean-face blend (per-model + global beta)               (+0.0029)
  6. per-image adaptive blend `pib` toward blurred-self, gamma=0.18          (+0.001)
"""
import numpy as np
import torch

import common as U
import config as C


# ---- Stage 0: OUR chroma correction ---------------------------------------
def desaturate(imgs, keep):
    """Shrink chroma toward luminance by (1-keep). keep=0 -> full gray."""
    if keep >= 0.999:
        return imgs
    lum = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
    lum = lum.unsqueeze(1)
    return (lum + keep * (imgs - lum)).clamp(0, 1)


def colorcast_fix(imgs):
    """Neutralise a low-freq per-channel colour cast: shift each channel so its
    batch mean equals the overall gray mean (removes model12's sigmoid tint)."""
    m = imgs.mean(dim=(0, 2, 3), keepdim=True)          # (1,3,1,1) per-channel mean
    gray = imgs.mean()                                  # global mean
    return (imgs - m + gray).clamp(0, 1)


# ---- Stage 4: global metric denoise ---------------------------------------
def nlm_denoise(imgs, h=0.05, patch=5, dist=6):
    """Edge-preserving NL-means. Per-image noise sigma is estimated dependency-free
    (std of the high-freq residual after a light blur) instead of skimage's
    estimate_sigma, which needs PyWavelets. Matches ff_build's h=max(0.05,0.85*s)."""
    from skimage.restoration import denoise_nl_means
    imgs = imgs.clamp(0, 1)
    resid = (imgs - U.gaussian_blur(imgs, sigma=1.0)).reshape(imgs.shape[0], -1)
    sig = resid.std(dim=1).cpu().numpy()               # per-image hi-freq noise proxy
    x = imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        s = float(sig[i])
        out[i] = denoise_nl_means(x[i], h=max(h, 0.85 * s), sigma=s,
                                  patch_size=patch, patch_distance=dist,
                                  channel_axis=-1, fast_mode=True)
    return torch.from_numpy(out).permute(0, 3, 1, 2).clamp(0, 1).float()


def metric_denoise(imgs):
    x = U.gaussian_blur(imgs, sigma=C.BLUR_SIGMA)       # blur first (+0.0259)
    x = nlm_denoise(x, h=C.NLM_H, patch=C.NLM_PATCH, dist=C.NLM_DIST)   # NLM on top
    return x.clamp(0, 1)


# ---- Stage 5: weak/dead-model mean-face blend -----------------------------
def mean_blend(imgs, mean_face, beta):
    if beta <= 0:
        return imgs
    return ((1 - beta) * imgs + beta * mean_face.unsqueeze(0)).clamp(0, 1)


# ---- Stage 6: per-image adaptive blend (pib) ------------------------------
def pib(imgs, gamma=0.18, sigma=1.0):
    """Blend each image toward its blurred copy in proportion to its NOISE level,
    where noise = low quality (rng_frac + 0.3*contrast - 2*tv). Grainy slots pulled
    hard toward smooth; clean slots stay sharp. gamma caps the blend."""
    if gamma <= 0:
        return imgs
    q = U.quality_score(imgs)
    lo, hi = torch.quantile(q, 0.10), torch.quantile(q, 0.90)
    noise = ((hi - q) / (hi - lo + 1e-9)).clamp(0, 1)    # low quality -> high noise
    alpha = (gamma * noise).view(-1, 1, 1, 1)
    blurred = U.gaussian_blur(imgs, sigma)
    return ((1 - alpha) * imgs + alpha * blurred).clamp(0, 1)


# ---- per-model driver ------------------------------------------------------
def postprocess_model(name, imgs, mean_face):
    idx = int(name.replace("model", ""))
    x = imgs.float().clamp(0, 1)
    # 0. OUR chroma correction (CNN only)
    if idx in C.DESATURATE:
        x = desaturate(x, C.DESATURATE[idx])
    if C.COLORCAST_FIX.get(idx):
        x = colorcast_fix(x)
    # 4. global metric denoise
    x = metric_denoise(x)
    # 5. weak/dead-model mean-face blend (per-model, else global)
    beta = C.PER_MODEL_MEAN_BETA.get(idx, C.GLOBAL_MEAN_BETA)
    mf = mean_face
    if idx in C.VIT_MODELS:                              # brighten ViT mean target
        mf = (mean_face - mean_face.mean() + C.VIT_MEAN_BRIGHTEN).clamp(0, 1)
    x = mean_blend(x, mf, beta)
    # 6. per-image adaptive blend
    x = pib(x, gamma=C.PIB_GAMMA)
    return x.clamp(0, 1)
