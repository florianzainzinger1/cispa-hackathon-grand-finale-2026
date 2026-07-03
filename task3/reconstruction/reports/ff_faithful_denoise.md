# Fundamental recon lever: FAITHFUL-RECON + noise-appropriate denoise

## ⛔ LEADERBOARD RESULT: REFUTED — submission_id 1453, score 0.3018, delta −0.0703 (best stays 0.3722)
Submitted `submission_ff.pt` (2026-07-03). It **regressed hard**. The faithful-recon+denoise approach
scored WORSE than blurry-c3 despite (a) lower top-128 subspace residual on all 4 MLPs and (b) visibly
more facial detail. **Lesson (now confirmed a 3rd time, after c4 net-zero and c5 −0.0045):** for this
SSIM metric, faithful-but-imperfect DETAIL loses to smooth BLUR. SSIM is low-pass biased and penalizes
structural error (even correct-identity error) more than it rewards recovered detail. **The subspace
residual measures gradient-faithfulness, NOT SSIM — do not use it to gate submissions.** The path to 0.48
is almost certainly EXACTNESS (verified verbatim originals / ground_truth.pt), not better denoising of
imperfect analytic recons. Everything below is the (refuted) hypothesis, kept for the record.

---


**Candidate:** `recon_work/outputs_ff/submission_ff.pt` (full 12-model, valid). Report:
`recon_work/outputs_ff/ff_report.json`. Build: `ff_build.py` + `job_ff.sh` (job 15402574, COMPLETED).
**Does NOT touch** the current best `test_nlm.pt` / c3 / current_best — own namespace `outputs_ff/`, prefix `ff_*`.

## The finding (why 0.3718 is stuck)
The 0.3718 best = **c3 (blurry/bank sources) + one Gaussian NLM (h=0.08)**. But Agent T's
`outputs_trap/rec_model*.pt` are **FAITHFUL** reconstructions — low top-128 gradient-subspace residual
(CNN 2/3/7/10/12 ≈ 0.025–0.062; MLP 1/4/5/8 ≈ 0.13–0.24) — i.e. they ARE the true images buried in
**model-specific noise**: fine Gaussian grain on MLPs, colored-blob impulse/clip on CNNs, haze on traps.
The best submission **threw that recovered detail away** by using blurry sources. The fleet's single
NLM-on-c3 was both the **wrong source** (blurry, not faithful) and the **wrong noise model** (Gaussian NLM
vs CNN impulse). This is a source-level reconstruction gap, NOT a metric tweak.

## The lever
Per-FAMILY denoise of the FAITHFUL recon, keeping detail:
- **MLP (1,4,5,8):** edge-preserving NLM, `h≈0.85·σ̂` (σ̂ = robust MAD-of-residual, no pywt).
- **CNN (2,3,7,10,12):** chroma-outlier clamp (replace pixels >0.35–0.45 from local median) → median(3) →
  light bilateral. Removes colored speckle without blurring.
- **Kept current-best:** model6 (rec residual ≈0.65 = too noisy), ViT 9/11 (no recoverable signal → bank).

## GT-blind validation — top-128 subspace residual (true images live in that subspace; lower = more faithful)
Denoise DECREASES residual = moves toward the true image; and beats the current best on all 4 MLPs:

| model | raw rec | **denoised (ff)** | current best 0.3718 |
|---|---|---|---|
| model1 | 0.191 | **0.095** | 0.116 |
| model4 | 0.186 | **0.097** | 0.122 |
| model5 | 0.212 | **0.104** | 0.126 |
| model8 | 0.131 | **0.157** | 0.189 |

model8's NLM slightly over-smoothed (resid rose vs raw rec but still < best) → lower its `h`.
Visual: MLP (model5) + trap-CNN (model10) are clean, detailed, recognizable (grids rendered, speckle gone).

## To confirm (only the leaderboard is truth — human submits)
Submit `outputs_ff/submission_ff.pt`. Expected to beat 0.3718 (4 MLP wins are residual- AND visually-confirmed;
5 CNN swaps are faithful-source + speckle-cleaned). If it regresses, safe fallback = swap ONLY the 4 MLPs into
the current best and keep CNNs as-is (MLP wins are the most certain). Next tuning: per-model `h`, and a stronger
CNN chroma model for model2.
