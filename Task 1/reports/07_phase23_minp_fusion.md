# Phase 2–3 findings — windows, p-values, min-p fusion

**Status:** DONE. `sub_002_minlr_fusion.jsonl` submitted — public LB **0.0319** (prev. 0.0009).

## The decisive discovery: boundary bleed dominates the FP tail

Naive min-p fusion (max evidence over centered/left/right windows, then max over schemes) UNDERPERFORMED TextSeal alone (val 0.0128 vs 0.0158 @0.1%FPR) — every added scheme brings its own clean tail (multiplicity cost). Inspecting Gumbel's top "false positives" showed they were **not noise**: validation_75 positions 810–819 sit directly BEFORE a true span starting at 820 — centered/right windows reach into the span and light up clean boundary tokens. The 0.1% FPR budget was being eaten almost entirely by span-adjacent clean tokens.

**Fix: per scheme, evidence = max over window sizes of min(left-window, right-window).** A token only scores high if BOTH sides look watermarked. Deep-span tokens qualify; boundary-adjacent clean tokens cannot.

| variant (validation) | TPR@0.1%FPR | TPR@1%FPR |
|---|---|---|
| max-all-windows fusion (4 schemes) | 0.0128 | 0.0713 |
| best subset without gumbel | 0.0158 | 0.0773 |
| **min(L,R) fusion (4 schemes)** | **0.0764** | **0.2241** |

Per-scheme after the fix (validation @0.1%): gumbel 0.0528, textseal 0.0416, kgw 0.0215, unigram 0.0029 — Gumbel went from "hurts fusion" to strongest scheme; its FP problem was 100% boundary bleed. Fusion now beats every single scheme. (Train @0.1% is only 0.0167 — train's clean tail contains nastier degenerate documents; the pooled 0.1% operating point has high subset variance, cf. public LB 0.0319.)

## Also tried, rejected

- **Empirical per-scheme tail calibration on train** (ECDF of clean evidence): train 0.0219 but validation **0.0000** — with ~47k clean tokens you cannot resolve p<2e-5; the empirical tail overfits catastrophically. Analytic p-values transfer; empirical tails don't. (Confirms strategy §9.1.)

## Pipeline

`predict_minp.py`: cache → per-scheme window features (binomial for green-lists γ=0.5/0.25, Gamma for gumbel var=1 / textseal var=0.5, v1-dedup validity masks) → min(L,R) per size → max over sizes → max over schemes → score = ev/(ev+20) ∈ [0,1). Evidence parquets cached per split for Phase 4/5 iteration.

## Phase 4 leads (ranked by expected value)

1. **Span-edge recovery**: min(L,R) sacrifices the first/last ~w tokens of true spans (val @1% jumped 3×, so mass sits just below the 0.1% threshold). An HMM posterior over per-token scores (not windows) should recover edges without re-opening boundary bleed.
2. Train's degenerate documents (train_23 etc.) still clog its tail — degenerate-token downweighting for green schemes beyond repeated-2grams.
3. Smaller windows (8) via min(L,R) for short spans; window-pair asymmetry (min(left_16, right_64)).
4. Optional: entropy weighting (Qwen forward pass) — Phase 5.

## Submissions so far

| id | file | public LB |
|---|---|---|
| 73 | sub_001_random_baseline | 0.0009 |
| 113 | sub_002_minlr_fusion | **0.0319** |
