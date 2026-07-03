"""Agent B — metric-tail lever: per-model LUMINANCE/CONTRAST match to a MEASURED
CelebA-64 reference, across all 12 models.

SSIM = l * c * s. The luminance term l is maximal when the batch mean equals the true
face mean; the contrast term c when the std matches. We MEASURE the reference per-channel
(mean, std) from the real CelebA top-200 gallery (12048 faces) — cross-checked against the
pipeline's own clean scored MLP faces so we never re-use the known-broken ~0.99 gallery — and
match EACH model's batch photometry to it.

GRAIN GUARD (important): the structure term s DOMINATES the contrast term here (blur/NLM won
+0.0397 by LOWERING contrast). So we do full luminance (mean) matching but only GENTLE,
capped contrast (std) matching — never amplifying std back toward the sharp gallery, which
would re-inject the grain we removed.

Produces on the confirmed nlm05 base (0.3722):
  test_lumonly.pt  : mean (luminance) match only, std frozen  [safest]
  test_lumatch.pt  : mean match + capped contrast (std scale in [0.9,1.1])
Nothing is submitted; submission_current_best.pt is untouched.
"""
import os
import torch

import utils as U
from build_submission import validate_submission

BASE = U.work_path("outputs_final", "submission_current_best.pt")   # == nlm05, 0.3722
CROPS = ["cc128", "cc148", "cc178", "r64"]
REF_CROP = "cc148"                                                   # HF/DDPM CelebA default


def measure_gallery(crop):
    g = torch.load(U.work_path("gallery", f"celeba_top200_{crop}.pt"),
                   weights_only=False, map_location="cpu")
    xf = (g["images"].float() / 255.0).reshape(-1, 3, 64 * 64)
    return xf.mean(dim=(0, 2)), xf.std(dim=(0, 2))


def measure_mlp(base):
    x = torch.cat([base[m].float() for m in ("model1", "model4", "model5")], 0)
    x = x.reshape(-1, 3, 64 * 64)
    return x.mean(dim=(0, 2)), x.std(dim=(0, 2))


def match(imgs, ref_mean, ref_std, std_lo, std_hi):
    """Per-channel batch photometry match. mean -> ref_mean fully; std -> ref_std
    scaled but clamped to [std_lo, std_hi] (grain guard)."""
    x = imgs.float()
    m = x.reshape(x.shape[0], 3, -1).mean(dim=(0, 2))
    s = x.reshape(x.shape[0], 3, -1).std(dim=(0, 2)).clamp_min(1e-4)
    scale = (ref_std / s).clamp(min=std_lo, max=std_hi)
    out = (x - m.view(1, 3, 1, 1)) * scale.view(1, 3, 1, 1) + ref_mean.view(1, 3, 1, 1)
    return out.clamp(0, 1).contiguous()


def build(base, rm, rs, std_lo, std_hi, tag):
    out = {m: match(base[m], rm, rs, std_lo, std_hi) for m in U.MODELS}
    errs = validate_submission(out)
    print(f"\n[{tag}] validate:", "OK" if not errs else errs)
    if errs:
        return
    path = U.work_path("outputs_final", f"test_{tag}.pt")
    torch.save(out, path)
    for m in U.MODELS:
        b, o = base[m].float(), out[m].float()
        print(f"   {m:8} mean {float(b.mean()):.3f}->{float(o.mean()):.3f}   "
              f"std {float(b.std()):.3f}->{float(o.std()):.3f}")
    print("   wrote", path)


def main():
    base = torch.load(BASE, weights_only=False, map_location="cpu")
    print("=== MEASURED references (per-channel mean / std) ===")
    for c in CROPS:
        mm, ss = measure_gallery(c)
        print(f"gallery {c:6}: mean {[round(x,3) for x in mm.tolist()]}  "
              f"std {[round(x,3) for x in ss.tolist()]}  (grey {float(mm.mean()):.3f})")
    mlpm, mlps = measure_mlp(base)
    print(f"pipeline MLP  : mean {[round(x,3) for x in mlpm.tolist()]}  "
          f"std {[round(x,3) for x in mlps.tolist()]}  (grey {float(mlpm.mean()):.3f})")

    rm, rs = measure_gallery(REF_CROP)
    if not (0.30 <= float(rm.mean()) <= 0.55):
        print(f"WARNING: gallery ref grey {float(rm.mean()):.3f} looks BROKEN -> using MLP-measured ref")
        rm, rs = mlpm, mlps
    print(f"\nUSING measured ref (crop {REF_CROP}): mean {[round(x,3) for x in rm.tolist()]}  "
          f"std {[round(x,3) for x in rs.tolist()]}")

    build(base, rm, rs, 1.0, 1.0, "lumonly")   # luminance only (std frozen)
    build(base, rm, rs, 0.9, 1.1, "lumatch")   # luminance + grain-guarded contrast
    print("\nDONE. Submit order (human, vs 0.3722): lumonly -> lumatch.")
    print("No local metric predicts the board; candidates only, submission_current_best untouched.")


if __name__ == "__main__":
    main()
