"""v3 clean face bank — a diverse, high-quality set of 128 real reconstructed
faces, used both as the ViT (9,11) answer and as the coverage fill for every
model (so no slot is ever noise).

Different from face_bank.py (which pools v1 selected_128 including its noise
padding and picks with KMeans). Here we pool the SHARP, noise-filtered, quality
-ranked reconstructions across the strong models and pick 128 by farthest-point
sampling for maximum diversity.
"""
import os
import sys
import torch

import utils as U
import v3_select as S

STRONG = ["model1", "model4", "model5", "model6", "model10", "model12"]


def clean_faces(name, tv_cut=0.20, per_model_cap=200):
    imgs, src = S.load_pool(name)
    if imgs is None:
        return torch.zeros(0, 3, 64, 64)
    q, st = S.quality_and_stats(imgs)
    real = S.is_real(st, tv_cut=tv_cut)
    idx = torch.nonzero(real, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return torch.zeros(0, 3, 64, 64)
    sub, sq = imgs[idx], q[idx]
    order = torch.argsort(sq, descending=True)[:per_model_cap]
    return sub[order]


def build(strong=STRONG, n=128, dedup_eps=0.03):
    pool = []
    for name in strong:
        f = clean_faces(name)
        if f.shape[0] > 0:
            pool.append(f)
    if not pool:
        raise RuntimeError("no clean faces for bank")
    imgs = torch.cat(pool, dim=0).clamp(0, 1)
    q, _ = S.quality_and_stats(imgs)
    feats = S.feat_vectors(imgs)
    # quality-ranked greedy dedup, then farthest-point top-up to n
    chosen, chosen_feats = S.greedy_dedup(imgs, q, feats, n=n, dedup_eps=dedup_eps)
    reps = [imgs[i] for i in chosen]
    if len(reps) < n:
        extra = S.farthest_point_fill(imgs, feats, chosen_feats, n - len(reps))
        for j in extra:
            reps.append(imgs[j])
    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    out_dir = U.work_path("outputs_v3")
    U.ensure_dir(out_dir)
    torch.save({"images": sel}, os.path.join(out_dir, "face_bank_v3.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "face_bank_v3_grid.png"), ncol=16)
    print(f"v3 bank: pooled {imgs.shape[0]} clean faces -> {sel.shape[0]} diverse "
          f"(from {len(strong)} strong models)")
    return sel


if __name__ == "__main__":
    build()
