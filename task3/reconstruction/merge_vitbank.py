"""Agent B — combined-pipeline builder for the ONE un-integrated safe lever:
the in-distribution ViT bank.

The confirmed current best (submission_current_best.pt == test_nlm05.pt, board 0.3722)
already bakes in every confirmed win EXCEPT a proper bank for the ViT slots: model9 and
model11 are byte-identical, a single generic recon-derived bank, and DARK (mean ~0.31 vs
the clean MLP faces ~0.44). Under Hungarian one-to-one SSIM the ViTs have no real recon to
lose to, so a cleaner / correctly-lit / in-distribution bank can only help or tie.

This builds THREE single-variable candidates on the nlm05 base (everything else untouched):
  test_vitbright.pt  : existing ViT bank, photometry matched to the pipeline's own clean
                       MLP faces (model1/4/5). Pure luminance/contrast fix, faces unchanged.
                       Lowest risk.
  test_vitgal148.pt  : ViT slots = diverse in-distribution bank sampled from the real
                       CelebA top-200 gallery (crop cc148), one clean face per distinct
                       identity, farthest-point diverse, light blur, photometry-matched.
  test_vitgal178.pt  : same but crop cc178 (framing hedge — true crop is unknown).

NOTHING is submitted and submission_current_best.pt is NOT overwritten. The human submits a
candidate to learn the real delta (no local metric predicts the board — the ff episode proved
proxies mislead). Ranked submit order: vitbright (safest) -> vitgal148 -> vitgal178.
"""
import os
import torch
import torch.nn.functional as F

import utils as U
from build_submission import validate_submission
from postprocess import gaussian_blur

BASE = U.work_path("outputs_final", "submission_current_best.pt")   # == nlm05, 0.3722
GAL = lambda crop: U.work_path("gallery", f"celeba_top200_{crop}.pt")
VIT = ["model9", "model11"]
N = U.N_IMAGES


def ref_photometry(base):
    """Per-channel (mean,std) of the pipeline's own clean scored MLP faces (1/4/5)."""
    x = torch.cat([base[m].float() for m in ("model1", "model4", "model5")], 0)
    x = x.reshape(-1, 3, 64 * 64)
    return x.mean(dim=(0, 2)), x.std(dim=(0, 2)).clamp_min(1e-4)


def moment_match(imgs, ref_mean, ref_std, std_cap=1.2, std_floor=0.85):
    m = imgs.reshape(imgs.shape[0], 3, -1).mean(dim=(0, 2))
    s = imgs.reshape(imgs.shape[0], 3, -1).std(dim=(0, 2)).clamp_min(1e-4)
    scale = (ref_std / s).clamp(min=std_floor, max=std_cap)
    out = (imgs - m.view(1, 3, 1, 1)) * scale.view(1, 3, 1, 1) + ref_mean.view(1, 3, 1, 1)
    return out.clamp(0, 1)


def gallery_id_bank(crop, sigma=0.8, n=N):
    """One clean non-flipped face per distinct identity, farthest-point diverse to n."""
    g = torch.load(GAL(crop), weights_only=False, map_location="cpu")
    imgs = g["images"].float() / 255.0
    ids = g["celeb_id"]
    isflip = g.get("is_flip", torch.zeros(len(ids)))
    reps = []
    for uid in torch.unique(ids).tolist():
        sel = torch.nonzero((ids == uid) & (isflip == 0), as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            sel = torch.nonzero(ids == uid, as_tuple=False).squeeze(1)
        reps.append(imgs[int(sel[0])])
    reps = torch.stack(reps)                                # (~200,3,64,64)
    # farthest-point sampling in 16x16 gray feature space for max diversity
    f = F.interpolate(reps.mean(1, keepdim=True), size=(16, 16),
                      mode="bilinear", align_corners=False).reshape(reps.shape[0], -1)
    picked = [0]
    mind = torch.cdist(f, f[[0]]).squeeze(1)
    while len(picked) < min(n, reps.shape[0]):
        j = int(torch.argmax(mind))
        picked.append(j)
        mind = torch.minimum(mind, torch.cdist(f, f[[j]]).squeeze(1))
    bank = reps[picked[:n]]
    if sigma > 0:
        bank = gaussian_blur(bank, sigma=sigma).clamp(0, 1)
    return bank


def _save(base, new_vit, tag, ref):
    out = {k: v.float().contiguous() for k, v in base.items()}
    vit = moment_match(new_vit.float(), *ref)
    for m in VIT:
        out[m] = vit.clone().contiguous()
    errs = validate_submission(out)
    path = U.work_path("outputs_final", f"test_{tag}.pt")
    print(f"\n[{tag}] validate:", "OK" if not errs else errs)
    if errs:
        return
    torch.save(out, path)
    for m in U.MODELS:
        d = float((out[m].float() - base[m].float()).abs().mean())
        flag = "CHANGED" if d > 1e-6 else "same"
        print(f"   {m:8} {flag:8} mean {float(base[m].float().mean()):.3f}->{float(out[m].float().mean()):.3f}")
    print("   wrote", path)


def main():
    base = torch.load(BASE, weights_only=False, map_location="cpu")
    ref = ref_photometry(base)
    print("current best == nlm05; ViT slots identical:",
          float((base["model9"].float() - base["model11"].float()).abs().mean()) < 1e-9)
    print("photometry ref (mean/std):", [round(x, 3) for x in ref[0].tolist()],
          [round(x, 3) for x in ref[1].tolist()])

    # (a) brighten the EXISTING bank (faces unchanged, photometry only)
    _save(base, base["model9"].float(), "vitbright", ref)
    # (b) in-distribution gallery id-bank, two crop hypotheses
    _save(base, gallery_id_bank("cc148"), "vitgal148", ref)
    _save(base, gallery_id_bank("cc178"), "vitgal178", ref)
    print("\nDONE. Submit order (human, vs 0.3722): vitbright -> vitgal148 -> vitgal178.")
    print("No local metric predicts the board; these are candidates, not confirmed gains.")


if __name__ == "__main__":
    main()
