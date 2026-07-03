# Agent B — Metric + geometry squeeze (Levers L-F, L-B)

**Status:** NOT STARTED · **Owner prefix:** `mx_*.py` · **Outputs:** `recon_work/outputs_mx/`
**Owns report:** this file. Submits via Agent C only. **Every candidate changes exactly ONE variable.**

## The one question
Can correct global photometric (mean/std/gamma to a REAL CelebA-64 reference) or geometric (recenter/shift)
transforms squeeze free SSIM out of the current best (0.3722, `test_nlm05.pt`)?

## Plan (see AGENT_SETUP.md §4 L-F/L-B + §5 CHAT B)
- [ ] Build a CORRECT CelebA-64 photometric reference (per-channel mean/std/hist of real cc178→64 faces, mean ≈0.40). The prior mean/std test was INVALID (broken 99%-white gallery).
- [ ] `mx_photometric.py`: candidate = per-channel mean/std match. Separate candidate = gamma/contrast match. (Do not stack.)
- [ ] `mx_geometry.py`: one candidate each for {shift +1px, shift −1px, 2% center-zoom}.
- [ ] Cheap-confirm (likely tapped): light blur σ=0.5 stacked on nlm05; blur σ re-tuned on nlm05 base.

## Findings
_(per-candidate leaderboard deltas; declare "metric/geometry tapped" if all flat)_

## Handoff to Agent C
_(candidate file path + one-line hypothesis, one at a time)_
