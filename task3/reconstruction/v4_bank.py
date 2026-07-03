"""v4 face bank — a cleaner, more diverse 128-face manifold for the ViT models
(9, 11) and every fill slot.

Difference vs v3_bank: ONE conservative change — DROP model6 from the source
pool. Its analytic feature maps are corrupted (per-row feature TV median 0.34 vs
0.09 for model10/12), so its "clean" survivors are the noisiest faces the v3
bank pulled in. Everything else (filter, selection) is identical to v3_bank so
we keep the pristine MLP faces (model1/4/5), which are the best in the bank.

Selection is unchanged: quality-ranked greedy dedup + farthest-point top-up to
128 for maximum diversity.
"""
import os
import sys
import torch

import utils as U
import v3_select as S

# no model6 (corrupt features); pristine MLP + clean stride-1 CNN
STRONG = ["model1", "model4", "model5", "model10", "model12"]


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
    counts = {}
    for name in strong:
        f = clean_faces(name)
        counts[name] = int(f.shape[0])
        if f.shape[0] > 0:
            pool.append(f)
    if not pool:
        raise RuntimeError("no clean faces for v4 bank")
    imgs = torch.cat(pool, dim=0).clamp(0, 1)
    q, _ = S.quality_and_stats(imgs)
    feats = S.feat_vectors(imgs)
    chosen, chosen_feats = S.greedy_dedup(imgs, q, feats, n=n, dedup_eps=dedup_eps)
    reps = [imgs[i] for i in chosen]
    if len(reps) < n:
        extra = S.farthest_point_fill(imgs, feats, chosen_feats, n - len(reps))
        for j in extra:
            reps.append(imgs[j])
    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    out_dir = U.work_path("outputs_v4")
    U.ensure_dir(out_dir)
    torch.save({"images": sel}, os.path.join(out_dir, "face_bank_v4.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "face_bank_v4_grid.png"), ncol=16)
    print(f"v4 bank: per-model clean counts={counts}")
    print(f"v4 bank: pooled {imgs.shape[0]} clean faces -> {sel.shape[0]} diverse")
    return sel


if __name__ == "__main__":
    build()
