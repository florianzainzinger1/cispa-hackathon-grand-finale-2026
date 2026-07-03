"""v3 selection — coverage-complete, diversity-first assembly of 128 images.

Distinct from v1 (KMeans cluster -> quality-weighted ROBUST MEAN, which blurs,
then NOISE-pads any shortfall) and v2 (confident single-sample clusters blended
with v1). Here we:

  1. Pool per-model candidates (refined > conv_invert > analytic), keeping the
     SHARP originals (no averaging).
  2. Drop noise: a candidate is "real" only if finite, mostly in-range, and its
     total-variation is below a face/noise cut (RGB noise has TV ~0.33; faces
     ~0.05-0.15). This is what removes the ~50% garbage slots on CNN models.
  3. Quality-rank the survivors, then GREEDY-DEDUP select up to 128 (skip any
     near-duplicate of an already chosen image) -> sharp + diverse, which is
     what the Hungarian one-to-one SSIM metric rewards.
  4. Fill any shortfall with the most DIVERSE faces from a real-face bank via
     farthest-point sampling -> every one of the 128 slots is a real face,
     NEVER noise.

Login-node safe (numpy/torch only).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

import utils as U
from postprocess import adaptive_denoise


# ----------------------------------------------------------------------------
# features + quality
# ----------------------------------------------------------------------------
def feat_vectors(imgs, size=16):
    """Downsampled gray feature for dedup / farthest-point diversity."""
    g = imgs.mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g.reshape(g.shape[0], -1)


def quality_and_stats(imgs):
    st = U.image_stats(imgs)
    contrast = torch.clamp(st["std"] / 0.25, max=1.0)
    q = st["rng_frac"] + 0.3 * contrast - 2.0 * st["tv"]
    q = torch.where(st["finite"], q, torch.full_like(q, -1e9))
    return q, st


def is_real(st, tv_cut=0.20, rng_cut=0.55):
    """Boolean mask: candidate looks like a face, not RGB noise."""
    return st["finite"] & (st["tv"] < tv_cut) & (st["rng_frac"] > rng_cut)


# ----------------------------------------------------------------------------
# candidate pool loading
# ----------------------------------------------------------------------------
def load_pool(name):
    """Return (images, source): the per-model reconstruction pool.

    NOTE: the gradient-matching `refined.pt` is deliberately NOT used — at
    batch-128 it converges to a high-cosine NOISE solution (verified: cos 0.98
    yet pure RGB/color noise for MLP/CNN/ViT alike). We use only the analytic
    attack outputs, which actually contain faces:
      * selected_128 (v1's curated per-model set) — a good base, esp. MLP.
      * conv_invert (CNN) — the full inverted pool (more clean slots than v1 kept).
      * analytic images_norm (MLP only) — sharp single-neuron faces with correct
        per-image min-max scaling (handles tanh/sigmoid MLPs the raw clamp broke).
    """
    d = U.work_path("candidates", name)
    parts, srcs = [], []

    sc = os.path.join(d, "selected_128.pt")
    if os.path.exists(sc):
        parts.append(torch.load(sc, weights_only=False)["images"].float()); srcs.append("selected_128")

    ci = os.path.join(d, "conv_invert_candidates.pt")
    if os.path.exists(ci):
        parts.append(torch.load(ci, weights_only=False)["images"].float()); srcs.append("conv_invert")

    ac = os.path.join(d, "analytic_candidates.pt")
    if os.path.exists(ac):
        blob = torch.load(ac, weights_only=False)
        # only MLP direct-image analytic gives real faces; CNN "feature_view" is not
        if blob.get("source") == "direct_image" and blob.get("images_norm") is not None:
            parts.append(blob["images_norm"].float()); srcs.append("analytic_norm")

    if not parts:
        return None, None
    imgs = torch.cat(parts, dim=0).clamp(0, 1)
    return imgs, "+".join(srcs)


def load_bank():
    # prefer the rebuilt clean/diverse v3 bank, fall back to v1's bank
    for p in (U.work_path("outputs_v3", "face_bank_v3.pt"),
              U.work_path("outputs", "face_bank.pt")):
        if os.path.exists(p):
            return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
    return None


# ----------------------------------------------------------------------------
# greedy dedup selection + farthest-point diversity fill
# ----------------------------------------------------------------------------
def greedy_dedup(imgs, q, feats, n=128, dedup_eps=0.08):
    """Add candidates in quality order, skipping near-duplicates. Returns list
    of chosen indices (into imgs)."""
    order = torch.argsort(q, descending=True).tolist()
    chosen, chosen_feats = [], []
    for i in order:
        if len(chosen) >= n:
            break
        fi = feats[i]
        if chosen_feats:
            d = torch.stack([(fi - cf).pow(2).mean().sqrt() for cf in chosen_feats])
            if float(d.min()) < dedup_eps:
                continue
        chosen.append(i); chosen_feats.append(fi)
    return chosen, chosen_feats


def farthest_point_fill(pool_imgs, pool_feats, have_feats, n_needed):
    """Greedily pick n_needed pool images maximizing min-distance to the growing
    selection (farthest-point sampling). Returns chosen pool indices."""
    if n_needed <= 0 or pool_imgs is None:
        return []
    sel_feats = list(have_feats)
    chosen = []
    # precompute min-dist of every pool item to the current selection
    P = pool_feats
    if sel_feats:
        S = torch.stack(sel_feats)
        mind = torch.cdist(P, S).min(dim=1).values
    else:
        mind = torch.full((P.shape[0],), 1e9)
    used = set()
    for _ in range(min(n_needed, P.shape[0])):
        j = int(torch.argmax(mind))
        if j in used:
            # all remaining are used/duplicates; break
            break
        used.add(j); chosen.append(j)
        dj = torch.cdist(P, P[j:j + 1]).squeeze(1)
        mind = torch.minimum(mind, dj)
        mind[j] = -1.0
    return chosen


def select_model(name, n=128, tv_cut=0.22, dedup_eps=0.03, verbose=True):
    imgs, src = load_pool(name)
    bank = load_bank()
    info = {"name": name, "pool_source": src}

    reps = []
    have_feats = []
    if imgs is not None:
        q, st = quality_and_stats(imgs)
        real = is_real(st, tv_cut=tv_cut)
        feats = feat_vectors(imgs)
        idx_real = torch.nonzero(real, as_tuple=False).squeeze(1)
        info["pool_size"] = int(imgs.shape[0])
        info["real_after_noise_filter"] = int(real.sum())
        if idx_real.numel() > 0:
            sub_imgs = imgs[idx_real]
            sub_q = q[idx_real]
            sub_feats = feats[idx_real]
            chosen_local, chosen_feats = greedy_dedup(sub_imgs, sub_q, sub_feats,
                                                      n=n, dedup_eps=dedup_eps)
            for li in chosen_local:
                reps.append(sub_imgs[li])
            have_feats = chosen_feats
    info["from_reconstruction"] = len(reps)

    # fill shortfall with diverse bank faces (NEVER noise)
    n_needed = n - len(reps)
    filled = 0
    if n_needed > 0 and bank is not None:
        bank_feats = feat_vectors(bank)
        fill_idx = farthest_point_fill(bank, bank_feats, have_feats, n_needed)
        for j in fill_idx:
            reps.append(bank[j])
        filled = len(fill_idx)
    info["from_bank"] = filled

    # last-resort: if STILL short (no bank), pad with slightly-perturbed picks
    while len(reps) < n:
        base = reps[len(reps) % max(1, len(reps))] if reps else torch.rand(3, 64, 64)
        g = torch.Generator().manual_seed(len(reps))
        reps.append((base + 0.02 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1))
    info["padded"] = max(0, n - info["from_reconstruction"] - info["from_bank"])

    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    # TV-gated denoise: only smooths noisy (high-TV) reconstruction slots toward
    # their underlying face; clean faces (low TV) and bank faces stay sharp.
    sel = adaptive_denoise(sel, tv_lo=0.12, tv_hi=0.28, max_alpha=0.6, sigma=1.0)
    out_dir = U.work_path("candidates_v3", name)
    U.ensure_dir(out_dir)
    torch.save({"images": sel, "info": info},
               os.path.join(out_dir, "selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_selected.png"), ncol=16)
    if verbose:
        print(json.dumps(info), flush=True)
    return sel, info


def main():
    names = sys.argv[1:] or U.MODELS
    allinfo = {}
    for name in names:
        try:
            _, info = select_model(name)
            allinfo[name] = info
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(allinfo, U.work_path("candidates_v3", "_select_summary.json"))
    print("wrote", U.work_path("candidates_v3", "_select_summary.json"))


if __name__ == "__main__":
    main()
