"""rt_montage.py - build a side-by-side montage: recon query (odd cols) next to
its retrieved gallery match (even cols), for visual identity verification.

Layout: an 8-row grid, each row shows 8 (recon, match) PAIRS = 16 tiles wide.
Sorted by match confidence (cosine) descending so the best matches are top-left.
Writes recon_work/outputs_rt/montage_model{m}_{variant}.png

Usage: python rt_montage.py <model_int> <variant> [n_pairs=64]
"""
import os
import sys
import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get('FL_WORK_ROOT', '/p/scratch/training2625/zainzinger1/1337/recon_work')
OUTDIR = os.path.join(ROOT, 'outputs_rt')


def load_recon(m):
    d = torch.load(os.path.join(ROOT, 'outputs_trap', f'rec_model{m}.pt'), map_location='cpu')
    return d['images'].float().clamp(0, 1)


def main():
    m = int(sys.argv[1])
    variant = sys.argv[2]
    n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    conf = np.load(os.path.join(OUTDIR, f'conf_model{m}_{variant}.npz'))
    pick = conf['pick']; cos = conf['cos']
    used_gallery = conf['used_gallery'] if 'used_gallery' in conf else np.ones(128, bool)
    g = torch.load(os.path.join(ROOT, 'gallery', 'celeba_top200_cc178.pt'), map_location='cpu')
    gimgs = g['images'].float() / 255.0
    recon = load_recon(m)
    sub = torch.load(os.path.join(OUTDIR, f'cand_model{m}_{variant}.pt'), map_location='cpu')
    out_block = sub[f'model{m}']    # what actually got submitted per slot

    order = np.argsort(-cos)[:n_pairs]      # best confidence first
    cols = 8
    rows = int(np.ceil(len(order) / cols))
    pad = 2
    tile = 64
    pairw = tile * 2 + pad
    W = cols * (pairw + pad) + pad
    H = rows * (tile + pad) + pad
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    def to_np(t):
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    for k, slot in enumerate(order):
        r = k // cols; c = k % cols
        y = pad + r * (tile + pad)
        x = pad + c * (pairw + pad)
        left = to_np(recon[slot])                 # the query recon
        right = to_np(out_block[slot])            # what got placed (gallery or fallback)
        canvas[y:y+tile, x:x+tile] = left
        canvas[y:y+tile, x+tile+pad:x+tile*2+pad] = right
    Image.fromarray(canvas).save(os.path.join(OUTDIR, f'montage_model{m}_{variant}.png'))
    print(f"montage saved: montage_model{m}_{variant}.png  ({len(order)} pairs, sorted by cos desc)")
    print(f"  top cos: {cos[order[:8]].round(3).tolist()}")
    print(f"  bottom cos of shown: {cos[order[-8:]].round(3).tolist()}")


if __name__ == '__main__':
    main()
