# Agent T — Trap-weight & exact analytic recovery (Lever L2)

**Status:** NOT STARTED
**Owner code prefix:** `trap_*.py` · **Outputs:** `recon_work/outputs_trap/`

## The one question
Which models have trap/imprint structure, and where present, can we recover VERBATIM images by
consecutive-bin differencing instead of the per-row ratio (averages) we've used so far?

## Tasks
- [ ] `trap_diag.py` — per-model first-layer diagnostics: weight/grad histograms, bias-gradient ordering, fraction of dead/negative-bias neurons, activation sparsity, singular spectrum. Flag ordered bin/imprint stacks. model10 = prime suspect (261 dead-bias neurons); check ALL 12.
- [ ] For flagged models — Robbing-the-Fed exact recovery: sort neurons by bias/threshold, `x_t = (gW[l] − gW[l+1]) / (gb[l] − gb[l+1])`; each non-colliding consecutive pair → one verbatim image; reject collisions (blended differences).
- [ ] Even for non-trap MLPs — test whether differencing single-sample-dominant neurons denoises better than averaging.
- [ ] Verify each recovered image by re-feeding (inferred label from `infer_labels.py`) → near-zero subspace residual.

## Findings
_(per-model table: trap? #verbatim-recovered, quality vs analytic-ratio faces)_

## Handoff to Agent B
_(which models have `trap_{model}.pt` ready and how many verbatim slots)_
