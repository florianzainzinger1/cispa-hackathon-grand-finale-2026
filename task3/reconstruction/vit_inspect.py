"""vit_inspect.py (agent2 / ViT specialist) — confirm the APRIL precondition and
map the leak structure of model9 & model11 BEFORE building the attack.

APRIL-analytic needs a LEARNABLE positional embedding: if pos_embed is a trained
Parameter, it has a gradient, and dL/dpos_embed = sum over the batch of the
per-sample token-input gradient dL/dz0. We check:
  * is 'pos_embed' present in the gradient dict (learnable) + its norm/rank
  * cls_token gradient (model9)
  * patch_embed.proj.weight gradient: reshape (dim, 3*p*p); is dim >= 3*p*p
    (square/invertible, model9) or dim < 3*p*p (lossy pseudo-inverse, model11);
    singular spectrum (is it the 'clean rank-128' AGENT_SETUP reports?)
  * first-block LN/qkv gradient presence (APRIL solves via the first layer)
  * head-bias gradient -> iLRG-style label-count hint
Login-node safe (tiny SVDs). Run: python -u vit_inspect.py
"""
import os
import sys
import torch

import utils as U

KEYS_OF_INTEREST = [
    "cls_token", "pos_embed",
    "patch_embed.proj.weight", "patch_embed.proj.bias",
    "blocks.0.norm1.weight", "blocks.0.norm1.bias",
    "blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.bias",
    "norm.weight", "fc_norm.weight", "head.weight", "head.bias",
]


def svprofile(M, tag):
    M = M.float()
    S = torch.linalg.svdvals(M)
    s0 = float(S[0]) + 1e-30
    e = float((S ** 2).sum()) + 1e-30

    def er(k):
        return round(float((S[:k] ** 2).sum()) / e, 4) if k <= S.numel() else None
    prof = {f"sv@{i}": round(float(S[i]) / s0, 4) for i in [1, 8, 16, 64, 127, 128]
            if i < S.numel()}
    return {
        "shape": tuple(M.shape), "rank_1pct": int((S > 0.01 * S[0]).sum()),
        "E@16": er(16), "E@64": er(64), "E@128": er(128),
        "sv_ratios": prof, "n_sv": int(S.numel()),
    }


def analyze(name):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    print(f"\n{'='*70}\n{name}: family={grad.get('family')} act={grad.get('activation')} "
          f"feat={grad.get('feature_shape')} B={grad.get('batch_size')}\n{'='*70}")

    # presence + norm of key gradients
    print("-- key gradients (present? shape, gradnorm) --")
    for k in KEYS_OF_INTEREST:
        if k in g and g[k] is not None:
            t = g[k].float()
            print(f"  {k:32s} shape={tuple(t.shape)} |grad|={t.norm():.4e}")
        elif k in g:
            print(f"  {k:32s} PRESENT but None")
    # anything else in grad dict not listed
    extra = [k for k in g if k not in KEYS_OF_INTEREST]
    print(f"  ... {len(g)} total grad tensors; sample extra keys: {extra[:6]}")

    # APRIL precondition: pos_embed learnable (has grad + in state)
    pe_learn = ("pos_embed" in g and g["pos_embed"] is not None)
    pe_in_state = "pos_embed" in state
    print(f"\n  APRIL precondition: pos_embed in_grad={pe_learn} in_state={pe_in_state}")
    if pe_learn:
        pg = g["pos_embed"].float().squeeze(0)          # (N, dim)
        print(f"  pos_embed grad {tuple(pg.shape)} |grad|={pg.norm():.4e} "
              f"per-token |grad| min/med/max="
              f"{pg.norm(dim=1).min():.3e}/{pg.norm(dim=1).median():.3e}/{pg.norm(dim=1).max():.3e}")
        print(f"  pos_embed grad spectrum: {svprofile(pg, 'pos')}")

    # patch_embed invertibility + spectrum
    pw = g.get("patch_embed.proj.weight")
    if pw is not None:
        dim = pw.shape[0]; p = pw.shape[-1]; ppix = 3 * p * p
        M = pw.float().reshape(dim, -1)                 # (dim, 3*p*p)
        print(f"\n  patch_embed: dim={dim} patch={p} patch_pixels={ppix} "
              f"-> {'SQUARE/INVERTIBLE' if dim >= ppix else 'LOSSY (dim<ppix, pinv only)'}")
        print(f"  patch_embed grad spectrum: {svprofile(M, 'pe')}")

    # head-bias -> label count hint (iLRG-style)
    hb = g.get("head.bias")
    if hb is not None:
        counts = (-U.N_IMAGES * hb.float()).clamp(min=0).round()
        print(f"\n  head.bias -> est present classes={int((counts>0).sum())} "
              f"sum={int(counts.sum())}/{U.N_IMAGES} (iLRG-ish label prior)")


def main():
    names = sys.argv[1:] or ["model9", "model11"]
    for n in names:
        try:
            analyze(n)
        except Exception as e:
            import traceback
            print(f"{n}: ERROR {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
