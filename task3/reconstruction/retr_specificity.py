"""retr_specificity.py (agent2 / retrieval) — is model10's cc178 low-residual signal
REAL members or a generic low-rank-face artifact?

model10's fc1 subspace flagged ~10-16 cc178 gallery faces at residual ~0.015 (vs 0.22
same-distribution baseline). A TRUE model10 member lies in span(model10's 128 private
images) -> low residual on model10, but is NOT a member of model6's (different) batch
-> NORMAL residual on model6 (same 8x64x64 feature space). A GENERIC low-rank face
would be low on BOTH. So:
  compare candidates' residual on model6 vs a random control's residual on model6.
  same  -> model10-SPECIFIC -> TRUE members.
  much lower -> generic -> artifact.
Also corroborate: best SSIM of each candidate to our grainy model10 recon (a true
member should have a grainy twin in our recon; elevated vs control).

Login-node safe (few hundred conv passes). python -u retr_specificity.py
"""
import os
import sys
import torch
import numpy as np

import utils as U
import models as M
from retr_lib import topk_right_subspace, subspace_residual
from skimage.metrics import structural_similarity as ssim

DEVICE = "cuda" if (os.environ.get("FL_DEVICE") == "cuda" and torch.cuda.is_available()) else "cpu"


def build(name):
    grad = U.load_grad(name); state = U.load_state(name); g = grad["gradients"]
    meta = {"family": "cnn", "activation": grad["activation"], "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state).to(DEVICE)
    V, _ = topk_right_subspace(g["fc1.weight"].float(), 128)
    return model, V


def feats(model, imgs, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], bs):
            outs.append(model.conv_feature(imgs[i:i+bs].to(DEVICE)).reshape(min(bs, imgs.shape[0]-i), -1).cpu())
    return torch.cat(outs, 0)


def ssim_pair(a, b):
    return ssim(a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0)


def best_ssim_to(recon, imgs, prek=24):
    R = recon.reshape(recon.shape[0], -1)
    out = []
    for i in range(imgs.shape[0]):
        q = imgs[i].reshape(-1)
        d = ((R - q).pow(2)).mean(1)
        cand = torch.argsort(d)[:prek]
        out.append(max(ssim_pair(imgs[i], recon[j]) for j in cand.tolist()))
    return np.array(out)


def main():
    crop = os.environ.get("RETR_CROP", "cc178")
    gal = torch.load(U.work_path("gallery", f"celeba_top200_{crop}.pt"), weights_only=False, map_location="cpu")
    imgs = gal["images"].float() / 255.0
    m10, V10 = build("model10")
    m6, V6 = build("model6")
    recon10 = torch.load(U.work_path("outputs_trap", "trap_model10.pt"), weights_only=False)["images"].float().clamp(0, 1)

    r10_all = subspace_residual(feats(m10, imgs), V10)
    order = torch.argsort(r10_all)
    cand = order[:64]
    torch.manual_seed(0)
    ctrl = order[torch.randint(2000, imgs.shape[0], (64,))]   # mid/high-residual random control

    def describe(tag, idx):
        sub = imgs[idx]
        r10 = r10_all[idx]
        r6 = subspace_residual(feats(m6, sub), V6)
        ss = best_ssim_to(recon10, sub)
        print(f"  {tag:10s} n={len(idx)}  resid10 mean={r10.mean():.4f} min={r10.min():.4f} | "
              f"resid6 mean={r6.mean():.4f} | bestSSIM_to_recon mean={ss.mean():.3f} max={ss.max():.3f}", flush=True)
        return dict(resid10_mean=float(r10.mean()), resid6_mean=float(r6.mean()),
                    ssim_mean=float(ss.mean()), ssim_max=float(ss.max()))

    print(f"=== model10 specificity @ {crop} (compare vs model6, same 8x64x64 feature space) ===", flush=True)
    c = describe("CANDIDATES", cand)
    k = describe("CONTROL", ctrl)
    # gallery-wide model6 baseline
    r6_all_med = float(subspace_residual(feats(m6, imgs[order[3000:3300]]), V6).median())
    print(f"  model6 gallery-median residual (non-members) ~ {r6_all_med:.4f}", flush=True)

    # verdict
    specific = c["resid6_mean"] > 0.6 * k["resid6_mean"] and c["resid10_mean"] < 0.5 * k["resid10_mean"]
    corrob = c["ssim_mean"] > k["ssim_mean"] + 0.03
    print(f"\n  candidates model10-SPECIFIC (resid6 ~ control, resid10 << control): {specific}", flush=True)
    print(f"  SSIM-to-recon corroboration (cand > control): {corrob} "
          f"({c['ssim_mean']:.3f} vs {k['ssim_mean']:.3f})", flush=True)
    verdict = "REAL_MEMBERS" if (specific or corrob) and c["resid10_mean"] < 0.06 else "ARTIFACT_or_WEAK"
    print(f"\n  VERDICT: {verdict}", flush=True)
    U.save_json({"crop": crop, "candidates": c, "control": k,
                 "model6_median_resid": r6_all_med, "specific": bool(specific),
                 "corroborated": bool(corrob), "verdict": verdict},
                U.work_path("reports", "retr", "specificity.json"))
    # save candidate faces for the record
    U.save_image_grid(imgs[cand[:32]], U.work_path("reports", "retr", "model10", f"candidates_{crop}.png"), ncol=8)


if __name__ == "__main__":
    main()
