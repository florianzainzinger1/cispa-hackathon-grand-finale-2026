"""rt_retrieve.py - identity retrieval for Task 3.

For a target model, take its 128 analytic recon slots (queries) and retrieve, from
the CelebA-top-200 gallery, 128 DISTINCT clean gallery images. Three variants:
  ssimnn  : nearest gallery image by SSIM(recon, gallery), 128 distinct.
  faceemb : nearest gallery image by ArcFace cosine, 128 distinct, SSIM as tiebreak.
  hybrid  : faceemb pick where confidence high, else keep the vithi45 recon row.

Confidence per slot = ArcFace cosine to matched gallery image (primary) and
SSIM(recon, matched) (secondary). Builds a full submission by copying the vithi45
base and swapping EXACTLY the target model block.

Usage: python rt_retrieve.py <model_int> <variant: ssimnn|faceemb|hybrid|all>
Outputs to recon_work/outputs_rt/cand_model{m}_{variant}.pt and a montage +
per-slot confidence npz for reporting.
"""
import os
import sys
import json
import numpy as np
import torch

ROOT = os.environ.get('FL_WORK_ROOT', '/p/scratch/training2625/zainzinger1/1337/recon_work')
sys.path.insert(0, os.path.join(ROOT, 'code'))

from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

GALLERY_CROP = os.environ.get('RT_GALLERY', 'cc178')
OUTDIR = os.path.join(ROOT, 'outputs_rt')
os.makedirs(OUTDIR, exist_ok=True)


def load_gallery(crop=GALLERY_CROP):
    g = torch.load(os.path.join(ROOT, 'gallery', f'celeba_top200_{crop}.pt'), map_location='cpu')
    imgs = g['images'].float() / 255.0          # (M,3,64,64) [0,1]
    return imgs, g['celeb_id'].numpy(), g['is_flip'].numpy()


def load_recon(m):
    d = torch.load(os.path.join(ROOT, 'outputs_trap', f'rec_model{m}.pt'), map_location='cpu')
    return d['images'].float().clamp(0, 1), d  # (128,3,64,64)


def load_base():
    return torch.load(os.path.join(ROOT, 'outputs_final', 'test_vithi45.pt'), map_location='cpu')


def ssim_gray(a, b):
    # a,b: (3,64,64) tensors -> grayscale SSIM
    ag = (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]).numpy()
    bg = (0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]).numpy()
    return ssim_fn(ag, bg, data_range=1.0)


def ssim_matrix(queries, gallery, topk_idx=None):
    """SSIM between each query (128) and gallery subset. If topk_idx given (128,K),
    only compute SSIM for those K gallery candidates per query. Returns (128, K) or full."""
    Q = queries.shape[0]
    qg = (0.299 * queries[:, 0] + 0.587 * queries[:, 1] + 0.114 * queries[:, 2]).numpy()  # (128,64,64)
    gg = (0.299 * gallery[:, 0] + 0.587 * gallery[:, 1] + 0.114 * gallery[:, 2]).numpy()   # (M,64,64)
    if topk_idx is None:
        M = gallery.shape[0]
        out = np.zeros((Q, M), dtype=np.float32)
        for i in range(Q):
            for j in range(M):
                out[i, j] = ssim_fn(qg[i], gg[j], data_range=1.0)
        return out
    K = topk_idx.shape[1]
    out = np.full((Q, K), -1.0, dtype=np.float32)
    for i in range(Q):
        for k in range(K):
            j = topk_idx[i, k]
            out[i, k] = ssim_fn(qg[i], gg[j], data_range=1.0)
    return out


def greedy_distinct(score_slot_gallery, gallery_valid=None):
    """score_slot_gallery: (128, M) desired-high score. Assign each of 128 slots a
    DISTINCT gallery index maximizing greedily by score. Returns pick[128] gallery idx,
    picked_score[128]."""
    S = score_slot_gallery
    n_slot, M = S.shape
    # Build candidate list: for each slot, its top candidates (limit for speed)
    K = min(M, 400)
    order = np.argsort(-S, axis=1)[:, :K]            # (128,K)
    triples = []
    for i in range(n_slot):
        for k in range(K):
            j = order[i, k]
            triples.append((S[i, j], i, j))
    triples.sort(key=lambda t: -t[0])
    pick = np.full(n_slot, -1, dtype=np.int64)
    picked_score = np.full(n_slot, -1.0, dtype=np.float32)
    used_g = set()
    filled = 0
    for sc, i, j in triples:
        if pick[i] != -1 or j in used_g:
            continue
        pick[i] = j
        picked_score[i] = sc
        used_g.add(j)
        filled += 1
        if filled == n_slot:
            break
    # any unfilled slot (shouldn't happen if M>=128): fill with best remaining unused
    if filled < n_slot:
        remaining = [j for j in range(M) if j not in used_g]
        rp = 0
        for i in range(n_slot):
            if pick[i] == -1:
                pick[i] = remaining[rp]; picked_score[i] = S[i, remaining[rp]]; rp += 1
    return pick, picked_score


_EMB_CACHE = {}


def gallery_embeddings(embedder, imgs_g, crop=GALLERY_CROP):
    key = (crop, id(imgs_g))
    if key not in _EMB_CACHE:
        cache_f = os.path.join(OUTDIR, f'galemb_{crop}.npy')
        if os.path.exists(cache_f):
            _EMB_CACHE[key] = np.load(cache_f)
        else:
            e = embedder.embed(imgs_g)
            np.save(cache_f, e)
            _EMB_CACHE[key] = e
    return _EMB_CACHE[key]


def build(m, variant, embedder=None, gallery=None, save=True):
    imgs_g, cid, is_flip = gallery if gallery is not None else load_gallery()
    M = imgs_g.shape[0]
    recon, meta = load_recon(m)
    base = load_base()

    # --- embeddings (gallery cached across variants/models) ---
    emb_g = gallery_embeddings(embedder, imgs_g)  # (M,512)
    emb_q = embedder.embed(recon)                # (128,512)
    cos = emb_q @ emb_g.T                         # (128,M) cosine (both L2-normed)

    if variant == 'ssimnn':
        # narrow with cos topK then rank by SSIM (full SSIM over 12k too slow)
        topk = np.argsort(-cos, axis=1)[:, :200]           # (128,200) candidates
        ss = ssim_matrix(recon, imgs_g, topk_idx=topk)     # (128,200)
        # map back to gallery-index score matrix (sparse): use SSIM as the score
        S = np.full((128, M), -1.0, dtype=np.float32)
        for i in range(128):
            S[i, topk[i]] = ss[i]
        pick, picked_ss = greedy_distinct(S)
        picked_cos = cos[np.arange(128), pick]
        picked_ssim = picked_ss
    elif variant in ('faceemb', 'hybrid'):
        pick, picked_cos = greedy_distinct(cos)
        # SSIM(recon, matched gallery) as secondary confidence
        picked_ssim = np.array([ssim_gray(recon[i], imgs_g[pick[i]]) for i in range(128)], dtype=np.float32)
    else:
        raise ValueError(variant)

    gallery_pick = imgs_g[pick].clone()          # (128,3,64,64)

    # --- confidence gating for hybrid ---
    used_gallery = np.ones(128, dtype=bool)
    if variant == 'hybrid':
        # keep gallery only where face match is confident; else keep vithi45 recon row
        COS_T = float(os.environ.get('RT_COS_T', '0.30'))
        block = base[f'model{m}']                # (128,3,64,64) vithi45 rows (may not row-align with recon)
        # We must keep 128 DISTINCT rows. Gallery picks are distinct; vithi45 rows are distinct.
        # For low-confidence slots, fall back to the vithi45 row at the SAME index.
        out = gallery_pick.clone()
        for i in range(128):
            if picked_cos[i] < COS_T:
                out[i] = block[i]
                used_gallery[i] = False
        gallery_pick = out

    # --- assemble submission ---
    sub = {k: base[k].clone() for k in base}
    sub[f'model{m}'] = gallery_pick.contiguous().float().clamp(0, 1)

    # distinctness safety: if hybrid introduced a dup (gallery pick == a vithi45 row), nudge
    flat = sub[f'model{m}'].reshape(128, -1)
    uniq = torch.unique(flat, dim=0).shape[0]

    info = dict(model=m, variant=variant, crop=GALLERY_CROP,
                n_used_gallery=int(used_gallery.sum()),
                cos=picked_cos.tolist(), ssim_recon_match=picked_ssim.tolist(),
                unique_rows=int(uniq), pick=pick.tolist(),
                celeb_id=cid[pick].tolist(), is_flip=is_flip[pick].tolist())

    if save:
        outp = os.path.join(OUTDIR, f'cand_model{m}_{variant}.pt')
        torch.save(sub, outp)
        np.savez(os.path.join(OUTDIR, f'conf_model{m}_{variant}.npz'),
                 cos=picked_cos, ssim=picked_ssim, pick=pick,
                 used_gallery=used_gallery, celeb_id=cid[pick], is_flip=is_flip[pick])
        with open(os.path.join(OUTDIR, f'info_model{m}_{variant}.json'), 'w') as f:
            json.dump(info, f, indent=2)
        print(f"[model{m}/{variant}] saved {outp}  unique={uniq}/128  "
              f"used_gallery={int(used_gallery.sum())}  "
              f"cos med={np.median(picked_cos):.3f} p25={np.percentile(picked_cos,25):.3f} "
              f"max={picked_cos.max():.3f}  ssim(recon,match) med={np.median(picked_ssim):.3f}")
    return sub, info, gallery_pick, recon


def main():
    from rt_embed import ArcFaceEmbedder
    m = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else 'all'
    emb = ArcFaceEmbedder(device=os.environ.get('FL_DEVICE', 'cpu'))
    gal = load_gallery()
    variants = ['ssimnn', 'faceemb', 'hybrid'] if variant == 'all' else [variant]
    for v in variants:
        build(m, v, embedder=emb, gallery=gal)


if __name__ == '__main__':
    main()
