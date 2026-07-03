"""iLRG-style label-gated candidate validation on the pib2 base (lever: hallucination filter).

Idea (honest, weak-by-construction):
  The classifier head gradient leaks WHICH class labels were present in the batch
  (iDLG / GradInversion / iLRG rule: for softmax-CE, gb[c] = mean_n(p_{n,c} - 1[y_n=c]),
  so a class truly present gets a NEGATIVE bias-gradient push while an absent class gets a
  small POSITIVE one -> present-set = { c : gb[c] < tau }).
  We then forward each of the current pib2 128 slots through the white-box model and read
  its predicted class. A slot whose predicted class is NOT in the recovered present-set is
  a candidate HALLUCINATION (the reconstruction drifted to a face the batch never contained),
  so we DOWN-WEIGHT it by pulling it harder toward the global mean face -- exactly the pib2
  blend, but with an extra label-gated beta applied ONLY to flagged slots. Clean slots are
  left byte-for-byte at pib2, so with zero flags the output == pib2 (safe single-variable).

WHY IT IS WEAK (stated up front, per the board laws):
  * ~128 of 200 classes are present, so absent-set is only ~72 classes. A hallucinated slot
    has to land its ARGMAX inside that ~36% absent slice to be caught -> low recall.
  * We do not know the models' input normalisation; we forward pib2 in [0,1] (the recon
    convention). If training used mean/std normalisation the argmax is unreliable -> both
    false-positive and false-negative flags. Keep gamma modest so a mis-flagged GOOD slot is
    nudged, not destroyed.
  * ViT {9,11} and defended {6} carry no per-sample readout (pib2 fills them with a blend /
    diverse gallery), so EVERY slot would "flag" and pulling them to mean just de-diversifies
    against the diversity law -> these are SKIPPED by default. Coordinator can override.
  Therefore this is a hedge, not a breakthrough: expect at best a small +delta on the signal
  models, and the current best is fully preserved when the filter fires on nothing.
  Board is the only truth (law 1): ship ONE variant at a time and read the leaderboard.

Outputs (never overwrites current_best):
  outputs_final/test_lfhard.pt   -- hard gate  : eb = gamma_hard * 1[argmax not in present]
  outputs_final/test_lfsoft.pt   -- soft gate  : eb = gamma_soft * P(absent classes)
  reports/build/labelfilter_masks.pt   -- per-model {present_mask, pred, flag, keep_mask, p_absent}
  reports/build/labelfilter_masks.json -- human summary (present-set size, #flagged, ...)

Login-node safe: torch + numpy only (no skimage/torchvision). CPU forward of 128 imgs is cheap.
Run:  python build_labelfilter.py            # default APPLY = signal models
      FL_LF_APPLY=model1,model10 python build_labelfilter.py
      FL_LF_TAU=0.0 FL_LF_GAMMA_HARD=0.35 FL_LF_GAMMA_SOFT=0.5 FL_LF_TARGET=global python build_labelfilter.py
"""
import os
import sys
import json
import numpy as np
import torch

import utils as U
import infer_labels as IL
import models as MODELS
import vit as VIT
from build_submission import validate_submission


# ---------------------------------------------------------------------------
# config (all env-overridable so the coordinator can sweep without editing)
# ---------------------------------------------------------------------------
BASE_PATH = os.environ.get("FL_LF_BASE", U.work_path("outputs_final", "test_pib2.pt"))
OUT_HARD  = os.environ.get("FL_LF_OUT_HARD", U.work_path("outputs_final", "test_lfhard.pt"))
OUT_SOFT  = os.environ.get("FL_LF_OUT_SOFT", U.work_path("outputs_final", "test_lfsoft.pt"))
MASK_PT   = U.work_path("reports", "build", "labelfilter_masks.pt")
MASK_JSON = U.work_path("reports", "build", "labelfilter_masks.json")

# present-set threshold: standard GradInversion rule is present <=> gb < 0.
# A slightly POSITIVE tau makes the present-set LARGER -> flags FEWER slots -> safer.
PRESENT_TAU = float(os.environ.get("FL_LF_TAU", "0.0"))
GAMMA_HARD  = float(os.environ.get("FL_LF_GAMMA_HARD", "0.35"))   # extra beta for a flagged slot
GAMMA_SOFT  = float(os.environ.get("FL_LF_GAMMA_SOFT", "0.50"))   # scales P(absent) in [0,1]
BETA_CAP    = float(os.environ.get("FL_LF_BETA_CAP", "0.90"))     # match pib2's clamp ceiling
TARGET_MODE = os.environ.get("FL_LF_TARGET", "global")           # 'global' | 'per_model'

# Only models whose pib2 slots are genuine per-sample reconstructions.
# SKIP {6 defended, 9/11 ViT}: their slots are filler, so the filter is meaningless/harmful.
DEFAULT_APPLY = ["model1", "model4", "model5", "model8",   # MLP
                 "model2", "model3", "model7",              # CNN pooled
                 "model10", "model12"]                      # CNN full-res
_env_apply = os.environ.get("FL_LF_APPLY", "").strip()
APPLY = [m.strip() for m in _env_apply.split(",") if m.strip()] or DEFAULT_APPLY

N = U.N_IMAGES  # 128


# ---------------------------------------------------------------------------
# 1. present-label set from the head bias gradient (iLRG / GradInversion rule)
# ---------------------------------------------------------------------------
def present_label_set(name, tau=PRESENT_TAU):
    """Return (present_mask[C] bool, gb[C] float, ilrg_counts[C] int, info dict).

    Standard rule: for softmax cross-entropy the head-bias gradient is
        gb[c] = (1/N) * sum_n (p_{n,c} - 1[y_n = c]).
    Present classes are pushed negative (the -1[y=c] term dominates), absent
    classes stay slightly positive. present <=> gb[c] < tau (tau=0 = canonical).
    ilrg_counts[c] = round(-N * gb[c]) is the iLRG per-class multiplicity estimate.
    """
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    wkey, bkey = IL.head_keys(state, g)
    if bkey is None or g.get(bkey) is None:
        return None
    gb = g[bkey].float().reshape(-1)
    present_mask = gb < tau
    counts = (-N * gb).clamp(min=0).round().to(torch.int64)
    info = {
        "n_classes": int(gb.numel()),
        "n_present": int(present_mask.sum()),
        "n_present_strict_ilrg": int((counts > 0).sum()),   # gb < -0.5/N
        "count_sum": int(counts.sum()),                     # ~ should be near 128
        "tau": tau,
    }
    return present_mask, gb, counts, info


# ---------------------------------------------------------------------------
# 2. build the white-box model and forward the 128 pib2 slots
# ---------------------------------------------------------------------------
def build_target_model(name):
    """Return an eval() nn.Module for MLP/CNN/ViT from the state_dict + grad meta."""
    meta = U.load_grad(name)          # has family / activation / feature_shape
    state = U.load_state(name)
    fam = meta.get("family")
    if fam in ("mlp", "cnn"):
        return MODELS.build_model(name, meta, state)
    if fam == "vit":
        m, _missing, _unexpected = VIT.build_vit(name, state)
        return m
    raise NotImplementedError(f"{name}: unknown family {fam}")


@torch.no_grad()
def slot_logits(model, imgs):
    """Forward (128,3,64,64) in [0,1] -> (128, C) logits. Recon convention = [0,1] input.
    NOTE: training-time input normalisation is unknown; this is a documented weakness."""
    x = imgs.float().clamp(0, 1)
    out = model(x)
    return out.float()


def flags_from_logits(logits, present_mask):
    """Hard flag + soft hallucination score from per-slot logits.

    hard flag[i]      = 1 if argmax_c logits[i,c] is NOT in the present-set.
    p_absent[i]       = softmax(logits[i]) mass on ABSENT classes (continuous, in [0,1]).
    pred[i]           = argmax class id.
    """
    C = logits.shape[1]
    pm = present_mask
    if pm.numel() != C:               # defensive: pad/trim to logit width
        fixed = torch.zeros(C, dtype=torch.bool)
        fixed[: min(C, pm.numel())] = pm[: min(C, pm.numel())]
        pm = fixed
    pred = logits.argmax(dim=1)
    flag = ~pm[pred]                                  # predicted class is absent -> hallucination
    p = torch.softmax(logits, dim=1)
    p_absent = p[:, ~pm].sum(dim=1).clamp(0, 1)       # soft version of the same signal
    return flag, p_absent, pred


# ---------------------------------------------------------------------------
# 3. pull flagged slots harder toward the (global) mean face  -- reuse pib2 blend
# ---------------------------------------------------------------------------
def blend_toward(imgs, target, beta_vec, cap=BETA_CAP):
    """out_i = (1 - b_i)*imgs_i + b_i*target ; b_i = clamp(beta_vec_i, 0, cap).
    target: (3,64,64). Clean slots (beta 0) are returned byte-for-byte unchanged."""
    b = beta_vec.float().clamp(0, cap).view(-1, 1, 1, 1)
    t = target.float().view(1, 3, 64, 64)
    return ((1 - b) * imgs.float() + b * t).clamp(0, 1)


def ensure_unique(x):
    """Guarantee 128 UNIQUE rows (organizer requirement). Only exact-duplicate rows are
    perturbed by a tiny deterministic dither, so non-colliding slots are untouched."""
    flat = x.reshape(x.shape[0], -1).clone()
    for i in range(1, flat.shape[0]):
        tries = 0
        while (flat[:i] == flat[i]).all(dim=1).any() and tries < 4:
            gseed = torch.Generator().manual_seed(4242 + i + tries)
            flat[i] = (flat[i] + 1e-3 * torch.randn(flat.shape[1], generator=gseed)).clamp(0, 1)
            tries += 1
    return flat.reshape_as(x).clamp(0, 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    argv_models = [a for a in sys.argv[1:] if a.startswith("model")]
    apply = argv_models or APPLY

    base = torch.load(BASE_PATH, weights_only=False, map_location="cpu")
    base = {m: base[m].float().clamp(0, 1).contiguous() for m in U.MODELS}
    print(f"base = {BASE_PATH}  (models={len(base)}, apply={apply})", flush=True)

    # global mean face = single averaged face over all APPLY slots (pib2's universal hedge).
    if TARGET_MODE == "global":
        stacked = torch.cat([base[m] for m in apply], dim=0)   # (128*k, 3,64,64)
        global_mean = stacked.mean(dim=0)                      # (3,64,64)
    else:
        global_mean = None                                     # per-model computed below

    out_hard = {m: base[m].clone() for m in U.MODELS}
    out_soft = {m: base[m].clone() for m in U.MODELS}
    masks = {}
    summary = {}

    for name in apply:
        pls = present_label_set(name)
        if pls is None:
            print(f"{name:8} no head-bias grad -> skip (kept pib2)", flush=True)
            summary[name] = {"skipped": "no head grad"}
            continue
        present_mask, gb, counts, info = pls

        try:
            model = build_target_model(name)
            logits = slot_logits(model, base[name])
        except Exception as e:                                  # arch mismatch -> keep pib2, don't crash
            print(f"{name:8} forward failed ({type(e).__name__}: {e}) -> skip (kept pib2)", flush=True)
            summary[name] = {"skipped": f"forward failed: {e}"}
            continue

        flag, p_absent, pred = flags_from_logits(logits, present_mask)
        n_flag = int(flag.sum())

        target = global_mean if global_mean is not None else base[name].mean(dim=0)

        out_hard[name] = ensure_unique(
            blend_toward(base[name], target, GAMMA_HARD * flag.float()))
        out_soft[name] = ensure_unique(
            blend_toward(base[name], target, GAMMA_SOFT * p_absent))

        masks[name] = {
            "present_mask": present_mask.cpu(),
            "gb": gb.cpu(),
            "ilrg_counts": counts.cpu(),
            "pred": pred.cpu(),
            "flag": flag.cpu(),
            "keep_mask": (~flag).cpu(),
            "p_absent": p_absent.cpu(),
        }
        summary[name] = {
            **info,
            "n_flagged_hard": n_flag,
            "frac_flagged_hard": round(n_flag / N, 3),
            "p_absent_mean": round(float(p_absent.mean()), 3),
            "p_absent_max": round(float(p_absent.max()), 3),
            "gamma_hard": GAMMA_HARD, "gamma_soft": GAMMA_SOFT, "target": TARGET_MODE,
        }
        print(f"{name:8} present={info['n_present']:3d}/{info['n_classes']} "
              f"(strict iLRG {info['n_present_strict_ilrg']}, sum={info['count_sum']})  "
              f"flagged_hard={n_flag:3d}/{N}  p_absent_mean={float(p_absent.mean()):.3f}",
              flush=True)

    # validate both variants exactly like the organizer's checker
    for tag, sub, path in (("hard", out_hard, OUT_HARD), ("soft", out_soft, OUT_SOFT)):
        errs = validate_submission(sub)
        for m in U.MODELS:
            uniq = torch.unique(sub[m].reshape(N, -1), dim=0).shape[0]
            if uniq != N:
                errs.append(f"{m}: only {uniq}/{N} unique rows")
        if errs:
            print(f"[{tag}] VALIDATION ERRORS -> NOT saving:")
            for e in errs[:6]:
                print("   -", e)
            continue
        U.ensure_dir(os.path.dirname(path))
        torch.save({m: sub[m].to(torch.float32).contiguous() for m in U.MODELS}, path)
        print(f"[{tag}] saved {path}", flush=True)

    # persist masks + human summary
    U.ensure_dir(os.path.dirname(MASK_PT))
    torch.save(masks, MASK_PT)
    U.save_json(summary, MASK_JSON)
    print(f"masks -> {MASK_PT}\nsummary -> {MASK_JSON}", flush=True)

    tot_flag = sum(v.get("n_flagged_hard", 0) for v in summary.values())
    tot_slot = N * sum(1 for v in summary.values() if "n_flagged_hard" in v)
    print("\n=== label-filter summary ===")
    print(f"applied to {len([v for v in summary.values() if 'n_flagged_hard' in v])} models; "
          f"hard-flagged {tot_flag}/{tot_slot} slots "
          f"({(100.0*tot_flag/tot_slot if tot_slot else 0):.1f}%).")
    print("HONEST CAVEAT: present-set ~128/200 => absent slice ~36%, so recall is low; "
          "unknown input normalisation makes argmax noisy. Clean slots == pib2 exactly. "
          "Board-test test_lfhard.pt and test_lfsoft.pt ONE at a time vs 0.3888 (law 1).")


if __name__ == "__main__":
    main()
