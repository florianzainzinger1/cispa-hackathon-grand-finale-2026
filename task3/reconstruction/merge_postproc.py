"""Agent B — SSIM-aware post-processing (lever L5). Free, uniform, stacks on any source.

SSIM is low-pass-biased and shift-sensitive, and it punishes grain far more than mild
blur (c5 regressed by shipping grainier faces). So the cheap wins are: smooth only the
noisy slots, match the natural-face colour distribution, and (only when a reference
original exists, e.g. Agent R retrieval) shift/scale to maximise SSIM.

Everything is TOGGLEABLE PER MODEL and every transform is VERIFIED not to increase grain
(mean TV) before it is accepted — a transform that raises TV is reverted for that image.

Reused primitives: postprocess.adaptive_denoise / gaussian_blur (no torchvision).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

import utils as U
from postprocess import adaptive_denoise, gaussian_blur


# ---------------------------------------------------------------------------
# 1. TV-gated denoise (smooth only noisy slots) — thin wrapper over postprocess
# ---------------------------------------------------------------------------
def tv_gated_denoise(imgs, tv_lo=0.12, tv_hi=0.28, max_alpha=0.6, sigma=1.0):
    return adaptive_denoise(imgs.float(), tv_lo=tv_lo, tv_hi=tv_hi,
                            max_alpha=max_alpha, sigma=sigma)


def light_blur(imgs, sigma=0.6, k=5):
    """Uniform mild blur. Use sparingly — only helps very grainy sets."""
    return gaussian_blur(imgs.float(), sigma=sigma, k=k).clamp(0, 1)


# ---------------------------------------------------------------------------
# 2. per-channel mean/std match to a natural-face reference distribution
# ---------------------------------------------------------------------------
def channel_moments(imgs):
    """Per-channel (mean,std) over a batch. imgs:(N,3,64,64)."""
    x = imgs.float().reshape(imgs.shape[0], 3, -1)
    mean = x.mean(dim=(0, 2))                     # (3,)
    std = x.std(dim=(0, 2)).clamp_min(1e-4)       # (3,)
    return mean, std


def channel_moment_match(imgs, ref_mean, ref_std):
    """Shift/scale each channel so the batch matches the reference per-channel
    (mean,std), then clamp to [0,1]. Preserves within-batch variation."""
    imgs = imgs.float()
    m, s = channel_moments(imgs)
    m = m.view(1, 3, 1, 1); s = s.view(1, 3, 1, 1)
    rm = ref_mean.view(1, 3, 1, 1); rs = ref_std.view(1, 3, 1, 1)
    return (((imgs - m) / s) * rs + rm).clamp(0, 1)


def celeba64_reference(default=(0.5063, 0.4258, 0.3832), default_std=(0.265, 0.245, 0.240)):
    """Canonical aligned-CelebA per-channel (mean,std) at 64px in [0,1].
    Used when no live reference set is available. Agent R can override with real
    gallery stats once retrieval is live."""
    return (torch.tensor(default, dtype=torch.float32),
            torch.tensor(default_std, dtype=torch.float32))


def reference_from_clean_models(sub, clean=("model1", "model4", "model5")):
    """Derive a face colour reference from the cleanest MLP reconstructions in a
    submission (they are the most face-faithful). Falls back to CelebA constants."""
    present = [m for m in clean if m in sub]
    if not present:
        return celeba64_reference()
    allimgs = torch.cat([sub[m].float() for m in present], dim=0)
    return channel_moments(allimgs)


# ---------------------------------------------------------------------------
# 3. reference alignment (only when a per-slot reference original exists)
# ---------------------------------------------------------------------------
def _ssim(a, b):
    """Scalar SSIM between two (3,64,64) tensors in [0,1] via skimage."""
    from skimage.metrics import structural_similarity as sk_ssim
    A = a.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    B = b.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return float(sk_ssim(A, B, channel_axis=2, data_range=1.0))


def align_to_reference(imgs, refs, max_shift=3, scales=(1.0,)):
    """For each image, search integer shifts in [-max_shift,max_shift]^2 (and
    optional small scales) for the version that MAXIMISES SSIM to its reference.
    Only meaningful when `refs` are true originals (e.g. retrieved). Returns the
    aligned batch. If refs is None, returns imgs unchanged."""
    if refs is None:
        return imgs.float()
    imgs = imgs.float().clamp(0, 1)
    refs = refs.float().clamp(0, 1)
    out = imgs.clone()
    for i in range(imgs.shape[0]):
        best, best_s = imgs[i], _ssim(imgs[i], refs[i])
        for sc in scales:
            base = imgs[i:i + 1]
            if sc != 1.0:
                base = F.interpolate(base, scale_factor=sc, mode="bilinear",
                                     align_corners=False)
                base = _center_crop_or_pad(base, 64)
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    cand = torch.roll(base, shifts=(dy, dx), dims=(-2, -1))[0].clamp(0, 1)
                    s = _ssim(cand, refs[i])
                    if s > best_s:
                        best_s, best = s, cand
        out[i] = best
    return out


def _center_crop_or_pad(x, size):
    _, _, h, w = x.shape
    if h == size and w == size:
        return x
    if h >= size:
        top = (h - size) // 2
        x = x[:, :, top:top + size, :]
    else:
        pad = size - h
        x = F.pad(x, (0, 0, pad // 2, pad - pad // 2), mode="reflect")
    _, _, h, w = x.shape
    if w >= size:
        left = (w - size) // 2
        x = x[:, :, :, left:left + size]
    else:
        pad = size - w
        x = F.pad(x, (pad // 2, pad - pad // 2, 0, 0), mode="reflect")
    return x


# ---------------------------------------------------------------------------
# 4. safe apply: revert per-image transforms that increase grain (TV)
# ---------------------------------------------------------------------------
def apply_if_not_grainier(orig, transformed, tol=1e-4):
    """Keep the transformed image only where it does NOT increase per-image TV."""
    orig = orig.float(); transformed = transformed.float()
    tv0 = U.total_variation(orig)
    tv1 = U.total_variation(transformed)
    keep = (tv1 <= tv0 + tol).view(-1, 1, 1, 1)
    return torch.where(keep, transformed, orig).clamp(0, 1)


# per-model default post-processing config. Agent B toggles per source/decision.
DEFAULT_CONFIG = {
    # grainy CNN / noisy MLP -> denoise; clean MLP -> leave sharp; bank/ViT -> as-is.
    "model1": dict(denoise=False, moment=True),
    "model4": dict(denoise=False, moment=True),
    "model5": dict(denoise=False, moment=True),
    "model8": dict(denoise=True,  moment=True),
    "model6": dict(denoise=True,  moment=True),
    "model10": dict(denoise=True, moment=True),
    "model12": dict(denoise=True, moment=True),
    "model2": dict(denoise=True,  moment=True),
    "model3": dict(denoise=True,  moment=True),
    "model7": dict(denoise=True,  moment=True),
    "model9": dict(denoise=False, moment=False),   # bank faces already clean
    "model11": dict(denoise=False, moment=False),
}


def postprocess_model(name, imgs, ref_mean=None, ref_std=None, refs=None, config=None):
    """Apply the per-model post-processing pipeline, each step grain-guarded."""
    cfg = (config or DEFAULT_CONFIG).get(name, dict(denoise=True, moment=True))
    x = imgs.float().clamp(0, 1)
    if cfg.get("denoise"):
        x = apply_if_not_grainier(x, tv_gated_denoise(x))
    if cfg.get("moment") and ref_mean is not None:
        x = apply_if_not_grainier(x, channel_moment_match(x, ref_mean, ref_std))
    if refs is not None:
        x = align_to_reference(x, refs)            # only when retrieval refs exist
    return x.clamp(0, 1)


if __name__ == "__main__":
    # smoke test on the current best: report TV before/after per model.
    sub = torch.load(U.work_path("outputs_final", "submission_current_best.pt"),
                     weights_only=False)
    rm, rs = reference_from_clean_models(sub)
    print(f"reference mean={rm.tolist()} std={rs.tolist()}")
    print(f"{'model':8} {'tv_before':>10} {'tv_after':>10}")
    for name in U.MODELS:
        if name not in sub:
            continue
        before = float(U.total_variation(sub[name].float()).mean())
        pp = postprocess_model(name, sub[name], ref_mean=rm, ref_std=rs)
        after = float(U.total_variation(pp).mean())
        print(f"{name:8} {before:10.4f} {after:10.4f}")
