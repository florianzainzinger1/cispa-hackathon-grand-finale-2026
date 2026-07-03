# Agent A — Distribution-matched blurred bank (Lever L-A)

**Status:** NOT STARTED · **Owner prefix:** `bank_*.py` · **Outputs:** `recon_work/outputs_bank/`, `recon_work/gallery/`
**Owns report:** this file. Submits via Agent C only.

## The one question
Does an in-distribution (CelebA top-200, cc178→64), blurred bank out-score the generic diverse bank on the
PURE-FILL models (ViT 9 & 11 first, then model6/8/5 tails) under Hungarian one-to-one SSIM?

## Plan (see AGENT_SETUP.md §4 L-A + §5 CHAT A)
- [ ] Build CLEAN CelebA top-200 gallery at cc178→64 (+cc148 alt). Verify mean ≈0.40 (NOT the broken 99%-white `gallery/celeba_top200_*.pt`).
- [ ] Diverse 128-face subset (farthest-point / k-center), one face per identity where possible.
- [ ] Blur to SSIM optimum: σ=1.0 (primary), σ=0.8 and NLM-h0.05 variants → `outputs_bank/bank_vit_*.pt`.
- [ ] Candidate = current best with ONLY model9 & model11 swapped → hand to Agent C.
- [ ] On confirmed +: extend to model6/8/5 tails (one model per submission). On −: report generic bank near-optimal, stop.

## Findings
_(per-swap leaderboard deltas, which σ/variant won)_

## Handoff to Agent C
_(candidate file path + one-line hypothesis)_
