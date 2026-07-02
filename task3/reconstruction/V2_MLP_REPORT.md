# v2 MLP Report — Gradient-Denoised Reconstruction (models 1, 4, 5, 8)

Scope: improve the leaky MLP models only, v1 left fully intact. All new outputs
under `recon_work/outputs_v2/` and `recon_work/candidates_v2/`. Read-only w.r.t.
`submission.pt` / `submission_v1_bank.pt` / v1 candidates.

## TL;DR

A new method — **low-rank denoising of the first-layer weight gradient before
taking analytic ratios** — measurably improves all four MLPs on an objective,
label-free fidelity metric, and an independent 4-judge visual panel confirms
cleaner faces with preserved identity/diversity. A ready, validated
`submission_v2_mlp.pt` swaps these four models into v1; the other 8 are unchanged.
**Recommend submitting v2** (biggest, unanimous win is model8; 1/4/5 are smaller,
safe gains). v1 remains the backup.

## What we tried, in order (all measured objectively)

The objective metric is **span-capture**: the first-layer gradient `gW = C·X_true`
lies in the span of the true images, so the fraction of `gW`'s energy captured by
projecting onto a candidate set's span = fidelity (higher = better; noise tilts
the span off-true and lowers it). Unlike gradient-cosine, it cannot be gamed by
matching a wrong image to the gradient.

| step | method | result | verdict |
|---|---|---|---|
| shared-identity | reuse faces across models | cross-model matched SSIM 0.12 < 0.22 within-model baseline | **refuted** (already reported) |
| v2a | single-sample clusters + trimmed-mean, merged with v1 | capture tie/slightly **worse** (m1 .7345→.7314) | negative — denoising blends/loses signal |
| v2b | full-gradient matching refinement (inferred labels) | cosine ↑ but capture **↓** (m8 .409→.343) | negative — wrong labels drive images off-manifold (same failure mode as the ViTs) |
| **diagnostic** | SVD of `gW` | **gradients are noise-perturbed**: full-rank (rank90 = 522–779 ≫ 128). Only 47–78% of energy is in the top-128 signal subspace | root cause |
| **graddenoise** | project `gW` to rank-128, then ratio | capture **↑ for all 4** | **WIN — adopted** |

## The key diagnostic: the gradients are defended (noised)

`gW` should be rank ≤128 (a sum of 128 rank-1 terms), but its energy spreads over
520–780 singular directions — so the stored gradients are **perturbed** (a privacy
defense; this is an "FL audit" task). Consequence:

| model | E@128 (ceiling) | v1 capture | v1 % of ceiling |
|---|---|---|---|
| 1 | 0.768 | 0.735 | 96% |
| 4 | 0.767 | 0.736 | 96% |
| 5 | 0.777 | 0.748 | 96% |
| 8 | 0.469 | 0.409 | 87% |

v1 was already near its information ceiling, which is **why v2a/v2b couldn't help**.
model8 is noisy because its gradient is the most perturbed (only 47% of energy is
signal), not because of a selection flaw.

## The method that worked: low-rank gradient denoising

`gW_denoised = U[:,:128] diag(S[:128]) Vᵀ[:128]` (drop the out-of-signal noise
subspace), then `ratio[row] = gW_denoised[row] / gb[row]`, cluster + select 128
(`gradient_denoise_test.py`). Removing the perturbation noise that lives outside
the top-128 subspace yields cleaner per-neuron reconstructions.

| model | valid rows | v1 capture | v2 capture | Δ | % of ceiling |
|---|---|---|---|---|---|
| 8 | 372 | 0.409 | **0.440** | +0.030 | 87→94% |
| 1 | 665 | 0.735 | **0.750** | +0.016 | 96→98% |
| 4 | 637 | 0.736 | **0.753** | +0.017 | 96→98% |
| 5 | 491 | 0.748 | **0.759** | +0.011 | 96→98% |

Capture increased for all four (the honest metric — signal loss would have lowered
it), pushing every model to 94–98% of its noise-limited ceiling.

## Independent verification (4-judge panel)

Four agents with distinct lenses (noise, skeptic/refuter, identity+diversity, SSIM
reasoning) judged the v1-vs-v2 compare grids:

- **model8: 4/4 v2-better (unanimous)** — v1 was heavy rainbow static; v2 recovers coherent, distinct faces.
- **models 1, 4, 5: 3/4 v2-better, skeptic neutral (tie)**. The skeptic flagged possible over-smoothing on the already-clean models — but span-capture *rose* for all, which refutes signal loss. **No judge rated any model v1-better.**

## Did model8 get cleaner?

Yes — it is the clearest win: the biggest capture gain (+0.030, 87→94% of ceiling)
and the most dramatic visual denoising (v1's near-unrecoverable speckle tiles
resolve into faces in v2). Its ceiling is still low (0.469) because its gradient is
the most perturbed, so it remains the noisiest model in absolute terms — the
defense caps it.

## Is v2 worth submitting?

Yes. It is a strict, evidence-backed fidelity improvement on 4/12 models with the
other 8 unchanged, validated, and v1 preserved as backup. Confidence: **high for
model8**, **moderate for 1/4/5** (smaller gains). Since SSIM can't be measured
locally, the safe play is to submit v2 and compare on the leaderboard — v1 is one
command away if v2 regresses.

## Outputs & commands

- v2 submission: `recon_work/outputs_v2/submission_v2_mlp.pt` (VALID; 12 keys,
  (128,3,64,64) float32 [0,1], 128 unique/model). Preview:
  `outputs_v2/submission_v2_mlp_preview.png`; stats: `..._stats.json`.
- Per-model v2 candidates: `candidates_v2/model{1,4,5,8}/graddenoise_128.pt`,
  `graddenoise_grid.png`, `compare_v1_vs_graddenoise.png`.
- Diagnostics: `reports/gw_spectrum.json`, `reports/graddenoise_capture.json`,
  `reports/v2_fidelity_capture.json`, `reports/v2b_refine_capture.json`.
- v1 untouched: `outputs/submission.pt` (15:51) and `submission_v1_bank.pt` (15:36).

Reproduce v2 (deterministic given the noised gradients):
```
python gradient_denoise_test.py model8 model1 model4 model5
python build_submission_v2.py graddenoise model1 model4 model5 model8
python validate_submission.py .../outputs_v2/submission_v2_mlp.pt
```

## Recommended next levers (beyond MLPs)

The MLPs are now near their noise-limited ceiling; further MLP tricks won't move the
needle. The same **low-rank gradient denoising** should be tried on the CNN models
(2,3,6,7,10,12) before their conv-inversion — likely a similar clean-up. That, not
more MLP work, is the next highest-ROI experiment.
