"""Per-model configuration for the merged pipeline.

Family/activation/feature_shape are read from the gradient bundle at runtime; this
file holds the TUNING knobs — the teammate's 0.3888 params plus OUR three validated
levers (marked  <-- OURS).

The 0.3888 climb, for reference:
  c3 0.3325 -> +model2 fix 0.3413 -> +blur/NLM 0.3722 -> +ViT blend 0.3837
  -> +weak blend 0.3851 -> +pib g0.10 0.3878 -> +pib g0.18 0.3888
"""
import os

# ---- reconstruction quality / compute (raise on the GPU cluster) ----------
# conv-inversion steps per CNN candidate. 500 is the login-node default; the
# cluster job sets MP_CONV_STEPS=3000 for sharper full-res inversions.
CONV_STEPS = int(os.environ.get("MP_CONV_STEPS", "500"))
TOPK = int(os.environ.get("MP_TOPK", "256"))         # candidates per model
# OUR validated lever: seed the conv-inversion from the feature channel-mean
# (low-freq face guess) instead of random noise -> faster, cleaner convergence,
# esp. on the under-determined strided/pooled models (2,3,7). Round-trip +0.05..0.12.
UPSAMPLE_INIT = True

# ---- Stage 1: analytic recon ----------------------------------------------
# OUR lever #1: aggressive low-rank denoise of the MLP first-layer weight gradient
# BEFORE the ratio readout. The teammate truncated at k=128 (kept ~90 noise dirs ->
# board tie). We use the per-model 2x-noise-floor crossover (validated +0.007 on the
# 0.31 track). 0 disables. model8's SNR is half the others -> lowest k.
MLP_LOWRANK_K = {1: 35, 4: 35, 5: 37, 8: 14}          # <-- OURS

# OUR lever #2: model2 is a tanh CNN whose analytic readout is CLEAN rank-128; the
# teammate scored feature candidates with a relu-style featureness (penalises <0),
# which flags every legitimately-negative tanh feature as bad. We score against the
# tanh output range [-1,1]. (Validated +0.0062 vs treating model2 as dead.)
# Handled in reconstruct.featureness() via the activation; no knob needed here.

# ---- Stage 4: global metric denoise (teammate's biggest lever, +0.031) ----
BLUR_SIGMA = 1.0
NLM_H = 0.05
NLM_PATCH, NLM_DIST = 5, 6

# ---- Stage 5: weak/dead-model mean-face blend -----------------------------
# blend each slot toward the GLOBAL mean face by beta (hedges no-signal slots).
GLOBAL_MEAN_BETA = 0.20
PER_MODEL_MEAN_BETA = {9: 0.50, 11: 0.50, 3: 0.45, 6: 0.40}   # ViT + weak CNNs
VIT_MEAN_BRIGHTEN = 0.36        # brighten the ViT mean target 0.31 -> 0.36

# ---- Stage 6: per-image adaptive blend (pib, peak gamma) -------------------
PIB_GAMMA = 0.18                # unimodal sweep peak (0.10->0.3878, 0.18->0.3888)

# ---- OUR lever #3: CNN conv-nullspace chroma correction -------------------
# model10 keeps neon-chroma noise (sat=0.35); crush it to grayscale like clean
# model12. model12 has a low-freq sigmoid COLOR-CAST -> neutralise per-channel mean.
DESATURATE = {10: 0.0}                       # keep-chroma factor (0 = full gray)  <-- OURS
COLORCAST_FIX = {12: True}                    # neutralise per-channel mean bias    <-- OURS

# which models get the ViT/bank-only treatment (no analytic recon)
VIT_MODELS = {9, 11}

# noise-gate + selection (v3 defaults)
TV_CUT = 0.22
RNG_CUT = 0.55
DEDUP_EPS = 0.03
