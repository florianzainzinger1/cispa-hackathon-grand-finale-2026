"""Stage 3: turn a per-model candidate pool into exactly 128 diverse real faces.

Teammate's v3 selection (diversity-first, never-noise):
  1. noise gate is_real: finite & TV<cut & rng_frac>cut  (drops ~50% CNN garbage)
  2. quality-rank survivors
  3. greedy dedup in 16x16 gray space -> sharp AND diverse (Hungarian rewards distinct)
  4. fill shortfall with farthest-point-diverse faces from a real-face bank (never noise)
  5. TV-gated denoise (only high-TV slots move toward blur)
The face bank is built from OUR cleanest recoveries (MLP 1/4/5 + full-res CNN 10/12).
"""
import torch

import common as U
import config as C


def is_real(st, tv_cut, rng_cut):
    return st["finite"] & (st["tv"] < tv_cut) & (st["rng_frac"] > rng_cut)


def greedy_dedup(imgs, q, feats, n=128, eps=0.03):
    order = torch.argsort(q, descending=True).tolist()
    chosen, cfeats = [], []
    for i in order:
        if len(chosen) >= n:
            break
        fi = feats[i]
        if cfeats:
            d = torch.stack([(fi - cf).pow(2).mean().sqrt() for cf in cfeats])
            if float(d.min()) < eps:
                continue
        chosen.append(i); cfeats.append(fi)
    return chosen, cfeats


def farthest_point_fill(bank, bank_feats, have_feats, n_needed):
    if n_needed <= 0 or bank is None:
        return []
    P = bank_feats
    if have_feats:
        mind = torch.cdist(P, torch.stack(have_feats)).min(dim=1).values
    else:
        mind = torch.full((P.shape[0],), 1e9)
    used, chosen = set(), []
    for _ in range(min(n_needed, P.shape[0])):
        j = int(torch.argmax(mind))
        if j in used:
            break
        used.add(j); chosen.append(j)
        mind = torch.minimum(mind, torch.cdist(P, P[j:j + 1]).squeeze(1))
        mind[j] = -1.0
    return chosen


def adaptive_denoise(imgs, tv_lo=0.12, tv_hi=0.28, max_alpha=0.6, sigma=1.0):
    tv = U.total_variation(imgs)
    alpha = ((tv - tv_lo) / (tv_hi - tv_lo)).clamp(0, 1) * max_alpha
    blurred = U.gaussian_blur(imgs, sigma)
    a = alpha.view(-1, 1, 1, 1)
    return ((1 - a) * imgs + a * blurred).clamp(0, 1)


def build_face_bank(pools, strong=(1, 4, 5, 10, 12), n=128):
    """Diverse real-face bank from the cleanest models' pools (farthest-point)."""
    imgs = []
    for i in strong:
        p = pools.get(f"model{i}")
        if p is not None:
            st = U.image_stats(p)
            imgs.append(p[is_real(st, C.TV_CUT, C.RNG_CUT)])
    if not imgs:
        return None
    imgs = torch.cat(imgs, 0).clamp(0, 1)
    feats = U.feat_vectors(imgs)
    q = U.quality_score(imgs)
    seed = [int(torch.argmax(q))]
    idx = farthest_point_fill(imgs, feats, [feats[seed[0]]], n - 1)
    return imgs[seed + idx][:n]


def select_model(name, pool, bank, n=128):
    info = {"name": name}
    reps, have_feats = [], []
    if pool is not None:
        st = U.image_stats(pool)
        real = is_real(st, C.TV_CUT, C.RNG_CUT)
        idx_real = torch.nonzero(real, as_tuple=False).squeeze(1)
        info["pool"] = int(pool.shape[0]); info["real"] = int(real.sum())
        if idx_real.numel() > 0:
            sub = pool[idx_real]
            q = U.quality_score(sub); feats = U.feat_vectors(sub)
            chosen, have_feats = greedy_dedup(sub, q, feats, n=n, eps=C.DEDUP_EPS)
            reps = [sub[i] for i in chosen]
    info["from_recon"] = len(reps)
    if len(reps) < n and bank is not None:                 # farthest-point bank fill
        bfeats = U.feat_vectors(bank)
        fill = farthest_point_fill(bank, bfeats, have_feats, n - len(reps))
        reps += [bank[j] for j in fill]
    info["from_bank"] = n - info["from_recon"] if len(reps) >= n else len(reps) - info["from_recon"]
    while len(reps) < n:                                    # last resort (no bank)
        g = torch.Generator().manual_seed(len(reps))
        base = reps[len(reps) % max(1, len(reps))] if reps else torch.rand(3, 64, 64)
        reps.append((base + 0.02 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1))
    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    sel = adaptive_denoise(sel)                             # TV-gated (v3)
    return sel, info
