"""Face-PCA manifold denoiser (agent3, UNTESTED lever — not in APPROACHES_LEDGER).

Projects the signal-model reconstructions onto a low-dim CelebA eigenface subspace.
Removes grain (law 2: smooth wins) like blur/NLM, BUT keeps each image's per-image
low-frequency identity (law 3: diversity wins under Hungarian) — unlike pib2's blend
toward a single GLOBAL mean face. Distinct from the DEAD archetypal/SPA lever (which
used the gradient's own noisy vertices; this uses a CLEAN external CelebA basis) and
from NLM (spatial vs subspace). Builds NON-destructive single-variable candidates on
the current-best base. Local proxies don't track the board (law 1) -> board-test only.

Namespace: merge_*  ->  outputs_final/merge_pca_*.pt  (never overwrites current_best).
"""
import os
import sys
import torch
import torch.nn.functional as F

import utils as U
from build_submission import validate_submission

SIGNAL = ["model1", "model4", "model5", "model8", "model10", "model12"]
MLP = ["model1", "model4", "model5", "model8"]


def load_gallery():
    for name in ["celeba_top200_cc178.pt", "celeba_top200_cc148.pt", "celeba_top200_cc128.pt"]:
        fp = U.work_path("gallery", name)
        if os.path.exists(fp):
            g = torch.load(fp, weights_only=False)
            t = g["images"] if isinstance(g, dict) and "images" in g else g
            t = t.float()
            if t.max() > 1.5:
                t = t / 255.0
            if t.shape[-1] != 64:
                t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
            print(f"gallery {name}: {tuple(t.shape)} range[{t.min():.2f},{t.max():.2f}]")
            return t.clamp(0, 1)
    return None


def eigenbasis(imgs, k):
    X = imgs.reshape(imgs.shape[0], -1).float()
    mean = X.mean(0, keepdim=True)
    q = min(k + 16, X.shape[0] - 1, X.shape[1])
    _, _, V = torch.pca_lowrank(X, q=q, center=True, niter=6)   # V: (D, q) principal dirs
    return mean, V[:, :k].t().contiguous()       # (1,D), (k,D)


def project(imgs, mean, E, strength):
    X = imgs.reshape(imgs.shape[0], -1)
    coeff = (X - mean) @ E.t()
    proj = coeff @ E + mean
    return ((1 - strength) * X + strength * proj).reshape_as(imgs).clamp(0, 1)


def build(base, slots, mean, E, strength):
    sub = {m: base[m].clone().float().clamp(0, 1) for m in U.MODELS}
    for m in slots:
        if m in sub:
            sub[m] = project(sub[m], mean, E, strength)
    return sub


def main():
    base_path = U.work_path("outputs_final", "submission_current_best.pt")
    base = torch.load(base_path, weights_only=False)
    print(f"base = {base_path} (keys={len(base)})")
    gal = load_gallery()
    if gal is None:
        print("NO GALLERY — abort")
        return
    out_root = U.work_path("outputs_final")
    variants = [
        ("mlp", MLP, 90, 0.5),
        ("mlp", MLP, 90, 0.75),
        ("sig", SIGNAL, 90, 0.5),
        ("sig", SIGNAL, 120, 0.65),
    ]
    for group, slots, k, strength in variants:
        mean, E = eigenbasis(gal, k)
        sub = build(base, slots, mean, E, strength)
        errs = validate_submission(sub)
        tag = f"merge_pca_{group}_k{k}_s{int(strength*100)}"
        if errs:
            print(f"{tag}: INVALID {errs[:2]} — skip")
            continue
        torch.save({m: sub[m].to(torch.float32).contiguous() for m in U.MODELS},
                   os.path.join(out_root, f"{tag}.pt"))
        tvb = float(U.total_variation(base["model1"].float()).mean())
        tva = float(U.total_variation(sub["model1"].float()).mean())
        # coverage sanity: distinct faces preserved?
        Xf = F.normalize(sub["model1"].reshape(128, -1), dim=1)
        uniq = int((torch.linalg.svdvals(Xf) > 1e-2 * torch.linalg.svdvals(Xf)[0]).sum())
        print(f"{tag}: valid; model1 TV {tvb:.4f}->{tva:.4f}, eff_unique={uniq}")
    print("\nBuilt merge_pca_* candidates (single-variable vs current best). Board-test ONE at a time.")


if __name__ == "__main__":
    main()
