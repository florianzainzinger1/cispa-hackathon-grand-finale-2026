"""vit_build.py (agent2 / ViT) — final deliverable for models 9 & 11.

VERDICT (see reports/vit/STATUS.md): APRIL / analytic ViT recovery yields only a
single batch-mean, low-frequency aggregate at B=128 — NOT per-sample identity
(self-test: B=1 SSIM 1.0 EXACT, collapsing to best-per-sample SSIM ~0.19 at B=128;
self-consistency proxy WORSE than the bank for both models). ViT 9/11 are
information-starved (patch_embed gradient effective rank 16 / 31).

Therefore the best available 128-image set for these ViTs is the DIVERSE BANK.
This builder writes outputs_vit/vit_{9,11}.pt = the diverse bank (128 unique real
faces), overwriting the useless mean-face-x128 that the analytic attack saved.
The APRIL aggregate + eigen-patches remain saved under reports/vit/ as evidence.
"""
import os
import sys
import torch

import utils as U

BANK_SRCS = [("outputs_v3", "submission_v3.pt"),
             ("outputs_final", "submission_current_best.pt"),
             ("outputs", "submission.pt")]


def load_bank(name):
    for rel in BANK_SRCS:
        p = U.work_path(*rel)
        if os.path.exists(p):
            d = torch.load(p, weights_only=False)
            if name in d:
                return d[name].float().clamp(0, 1), "/".join(rel)
    return None, None


def build(name):
    bank, src = load_bank(name)
    if bank is None:
        print(f"{name}: NO bank source found; cannot build fallback", flush=True)
        return None
    imgs = bank.float().clamp(0, 1)
    assert tuple(imgs.shape) == (128, 3, 64, 64), f"bad shape {imgs.shape}"
    uniq = int(torch.unique(imgs.reshape(128, -1), dim=0).shape[0])
    finite = bool(torch.isfinite(imgs).all())
    out = U.work_path("outputs_vit", f"vit_{name[5:]}.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save({
        "images": imgs, "method": "bank_fallback", "bank_src": src,
        "unique": uniq, "finite": finite,
        "note": ("APRIL/analytic recovers only aggregate low-frequency at B=128 "
                 "(B=1 exact, best-per-sample SSIM ~0.19 at B=128; self-consistency "
                 "proxy worse than bank). ViT patch_embed grad eff-rank 16/31 -> "
                 "information-starved. Bank is the best 128-set; use it for 9/11."),
    }, out)
    print(f"{name}: wrote {out}  src={src} unique={uniq}/128 finite={finite}", flush=True)
    return {"name": name, "src": src, "unique": uniq}


def main():
    names = sys.argv[1:] or ["model9", "model11"]
    for n in names:
        build(n)


if __name__ == "__main__":
    main()
