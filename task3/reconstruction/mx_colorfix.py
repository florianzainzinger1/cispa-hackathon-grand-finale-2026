"""mx_colorfix.py — build low-freq color/brightness-corrected candidates for
model12 (sigmoid color cast) and model10 (dark + rainbow speckle) on the
vithi45 base. Swap EXACTLY ONE model per candidate; keep the other 11 byte-identical.

THE ONE LAW: fix LOW-frequency errors (color/brightness/contrast) WITHOUT adding grain.
No sharpening anywhere. All denoise is grain-removal only.
"""
import os, sys, copy
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means

WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
sys.path.insert(0, f"{WORK}/code")
import utils as U

# ---- CelebA reference statistics (from mx_stats.py on celeba_top200_cc178, /255) ----
REF_MEAN = np.array([0.5146, 0.4229, 0.3754], dtype=np.float32)   # R,G,B
REF_STD  = np.array([0.3061, 0.2837, 0.2809], dtype=np.float32)
REF_GMEAN = 0.4376

BASE = f"{WORK}/outputs_final/test_vithi45.pt"


def load_base():
    return torch.load(BASE, map_location="cpu", weights_only=False)


def to_np(t):
    # (N,3,H,W) tensor -> (N,H,W,3) float32 numpy in [0,1]
    return t.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy()


def to_t(a):
    # (N,H,W,3) numpy -> (N,3,H,W) float32 tensor clamped [0,1]
    return torch.from_numpy(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2).clamp(0, 1)


# ---------------- transform primitives ----------------

def channel_mean_shift(a, strength=1.0, ref_mean=REF_MEAN):
    """Additive per-channel offset so channel means move toward ref (gray-world WB).
    strength in [0,1]; 1.0 = full match. Pure low-freq: shifts luminance/hue, no contrast change."""
    cur = a.reshape(-1, 3).mean(0)
    delta = (ref_mean - cur) * strength
    return a + delta.reshape(1, 1, 1, 3)


def channel_meanstd_match(a, strength=1.0, ref_mean=REF_MEAN, ref_std=REF_STD, std_cap=1.3):
    """Per-channel mean+std match toward ref. std_cap limits contrast BOOST (avoid amplifying grain)."""
    flat = a.reshape(-1, 3)
    cur_m = flat.mean(0)
    cur_s = flat.std(0) + 1e-6
    tgt_s = cur_s + (ref_std - cur_s) * strength
    # never amplify std by more than std_cap (amplifying = amplifying grain)
    scale = np.minimum(tgt_s / cur_s, std_cap)
    tgt_m = cur_m + (ref_mean - cur_m) * strength
    out = (a - cur_m.reshape(1, 1, 1, 3)) * scale.reshape(1, 1, 1, 3) + tgt_m.reshape(1, 1, 1, 3)
    return out


def per_image_bright_match(a, target=REF_GMEAN, strength=1.0, max_gain=1.8):
    """Scale each image's brightness toward `target` mean (multiplicative, capped).
    Fixes per-image darkness without touching relative structure."""
    out = a.copy()
    for i in range(a.shape[0]):
        m = a[i].mean()
        if m < 1e-4:
            continue
        gain = 1.0 + (target / m - 1.0) * strength
        gain = min(gain, max_gain)
        out[i] = a[i] * gain
    return out


def desaturate(a, amount=0.3):
    """Pull colors toward per-pixel luminance. amount in [0,1]; kills oversaturation (a low-freq cast fix)."""
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    return a * (1 - amount) + lum * amount


def gauss_lowpass(a, sigma=0.5):
    """Mild per-image, per-channel Gaussian low-pass. Removes residual grain. No sharpening."""
    if sigma <= 0:
        return a
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for c in range(3):
            out[i, ..., c] = gaussian_filter(a[i, ..., c], sigma=sigma, mode="nearest")
    return out


def nlm(a, h=0.06):
    """Edge-preserving NLM denoise (the confirmed grain-killer). Per image, RGB channel-wise."""
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = denoise_nl_means(a[i], h=h, patch_size=5, patch_distance=6,
                                  channel_axis=-1, fast_mode=True)
    return out


def despeckle(a, thresh=0.18, med_size=3):
    """Remove rainbow SPECKLE: replace only pixels that deviate strongly from a local median
    (outliers), leaving the rest untouched. Targets isolated colored dots, not structure."""
    out = a.copy()
    for i in range(a.shape[0]):
        for c in range(3):
            ch = a[i, ..., c]
            med = median_filter(ch, size=med_size, mode="nearest")
            mask = np.abs(ch - med) > thresh
            out[i, ..., c][mask] = med[mask]
    return out


def clip01(a):
    return np.clip(a, 0.0, 1.0)


# ---------------- candidate builders ----------------

def save_candidate(model_key, new_imgs_t, out_name):
    """Copy base, swap ONLY model_key, save as outputs_mx/<out_name>."""
    sub = load_base()
    assert new_imgs_t.shape == (128, 3, 64, 64), new_imgs_t.shape
    sub[model_key] = new_imgs_t.float().clamp(0, 1).contiguous()
    outp = f"{WORK}/outputs_mx/{out_name}"
    torch.save(sub, outp)
    print(f"[saved] {outp}  {model_key} gmean={new_imgs_t.mean():.4f} "
          f"chan_mean={new_imgs_t.mean(dim=(0,2,3)).numpy().round(3)}")
    return outp


def build_all():
    sub = load_base()
    m12_v = to_np(sub["model12"])   # vithi45 model12 (already smooth)
    m10_v = to_np(sub["model10"])   # vithi45 model10 (dark, speckle)

    # raw trap recons
    r12 = torch.load(f"{WORK}/outputs_trap/rec_model12.pt", map_location="cpu", weights_only=False)["images"]
    r10 = torch.load(f"{WORK}/outputs_trap/rec_model10.pt", map_location="cpu", weights_only=False)["images"]
    m12_r = to_np(r12)
    m10_r = to_np(r10)

    outs = {}

    # ===== MODEL 12 =====
    # color1 MILD: gray-world white-balance (channel mean shift, full) on the smooth vithi45 base.
    #   Fixes residual hue cast + slight darkness, keeps existing contrast, no denoise needed (already smooth).
    a = channel_mean_shift(m12_v, strength=1.0)
    a = per_image_bright_match(a, target=REF_GMEAN, strength=0.5, max_gain=1.5)
    a = clip01(a)
    outs["cand_m12_color1.pt"] = ("model12", to_t(a))

    # color2 MEDIUM: mean+std match (WB + gentle contrast stretch toward CelebA) + mild low-pass to
    #   guarantee the stretch introduces no grain.
    a = channel_meanstd_match(m12_v, strength=0.7, std_cap=1.25)
    a = per_image_bright_match(a, target=REF_GMEAN, strength=0.4, max_gain=1.5)
    a = gauss_lowpass(a, sigma=0.5)
    a = clip01(a)
    outs["cand_m12_color2.pt"] = ("model12", to_t(a))

    # color3 FROM-TRAP: recover the sharper trap structure but AGGRESSIVELY kill the sigmoid
    #   oversaturation/grain: desaturate -> mean/std match (cap std so we DON'T amplify the huge trap std)
    #   -> NLM denoise -> mild low-pass. Tests if sharper structure survives de-graining.
    a = desaturate(m12_r, amount=0.35)
    a = channel_meanstd_match(a, strength=1.0, std_cap=1.0)  # cap 1.0 = never boost std (trap std huge)
    a = nlm(a, h=0.07)
    a = gauss_lowpass(a, sigma=0.6)
    a = per_image_bright_match(a, target=REF_GMEAN, strength=0.5, max_gain=1.6)
    a = clip01(a)
    outs["cand_m12_color3.pt"] = ("model12", to_t(a))

    # ===== MODEL 10 =====
    # clean1: vithi45 base (already smooth) — despeckle rainbow dots, lift brightness/contrast toward
    #   CelebA (it's very dark 0.276->~0.40), white-balance, mild low-pass.
    a = despeckle(m10_v, thresh=0.16, med_size=3)
    a = per_image_bright_match(a, target=0.40, strength=0.7, max_gain=1.7)
    a = channel_meanstd_match(a, strength=0.6, ref_mean=REF_MEAN, ref_std=REF_STD, std_cap=1.35)
    a = gauss_lowpass(a, sigma=0.4)
    a = clip01(a)
    outs["cand_m10_clean1.pt"] = ("model10", to_t(a))

    # clean2: trap real faces — despeckle, brightness/contrast match to ~0.44, WB, NLM+low-pass to
    #   remove the trap grain. Real-face structure with speckle+darkness fixed.
    a = despeckle(m10_r, thresh=0.16, med_size=3)
    a = per_image_bright_match(a, target=0.42, strength=0.7, max_gain=1.7)
    a = channel_meanstd_match(a, strength=0.6, std_cap=1.2)
    a = nlm(a, h=0.06)
    a = gauss_lowpass(a, sigma=0.4)
    a = clip01(a)
    outs["cand_m10_clean2.pt"] = ("model10", to_t(a))

    # save all
    paths = {}
    for name, (mk, t) in outs.items():
        paths[name] = save_candidate(mk, t, name)
    return paths


if __name__ == "__main__":
    os.makedirs(f"{WORK}/outputs_mx", exist_ok=True)
    build_all()
    print("DONE build_all")
