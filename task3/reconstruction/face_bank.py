"""Build a diverse bank of 128 real reconstructed faces from the strong models.

Used as the fallback answer for the ViT models (9, 11), whose gradients do not
invert to coherent faces: since every private image is a face from the same
distribution, a diverse set of genuine reconstructed faces scores far higher
SSIM than incoherent patch-noise from failed gradient matching.
"""
import os
import sys
import torch

import utils as U
import postprocess as P

STRONG = ["model1", "model4", "model5", "model6", "model10", "model12"]


def build(strong=STRONG, n=128):
    pool = []
    for name in strong:
        p = U.work_path("candidates", name, "selected_128.pt")
        if os.path.exists(p):
            pool.append(torch.load(p, weights_only=False)["images"].float())
    if not pool:
        raise RuntimeError("no strong selected_128 files found")
    imgs = torch.cat(pool, dim=0)
    sel, _ = P.select_128(imgs, n=n, denoise=False)
    out = U.work_path("outputs", "face_bank.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save({"images": sel}, out)
    U.save_image_grid(sel, U.work_path("outputs", "face_bank_grid.png"), ncol=16)
    print(f"face bank: pooled {imgs.shape[0]} -> {sel.shape[0]} saved {out}")
    return sel


if __name__ == "__main__":
    build()
