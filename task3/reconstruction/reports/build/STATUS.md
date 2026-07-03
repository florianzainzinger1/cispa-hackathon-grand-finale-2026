# Agent B — Build, measure, integrate, SSIM post-process (Lever L5 + assembly)

**Status:** CONTINUOUS · **Owns:** [SCOREBOARD.md](../SCOREBOARD.md) and `recon_work/outputs_final/submission_current_best.pt`
**Owner code prefix:** `merge_*.py`, `proxy_*.py` · **Outputs:** `recon_work/outputs_final/`

## Job
Keep an always-valid best submission, measure candidates without ground truth, squeeze free SSIM.

## Tasks
- [ ] `proxy_score.py` — no-GT signals: (a) gradient self-consistency (re-feed 128 imgs w/ inferred labels → cosine vs observed), (b) top-128 subspace residual, (c) diversity/coverage, (d) grain/TV. Per-model, per-source scoreboard.
- [ ] `merge_postproc.py` — per-image mild denoise + light Gaussian blur (σ 0.5–0.8), per-channel mean/std match to CelebA-64, optional ±3px/scale alignment sweep. Toggleable per model; never increases grain.
- [ ] `merge_build.py` — per model pick best source among {c3 baseline, outputs_retr, outputs_trap, outputs_bss, outputs_gp} via the proxy scoreboard; enforce exactly 128 distinct finite [0,1] (128,3,64,64); fill shortfalls with diverse bank (never noise). Always leave valid `submission_current_best.pt` + manifest of which model came from which source.
- [ ] Run `validate_submission.py` on every build. Keep the [SCOREBOARD.md](../SCOREBOARD.md) submission log current from human-reported deltas. Never burn a 5-min cooldown on an unverified change.

## Current recommendation for next submission
_(fill in: which model swaps to include, expected delta, rationale)_
