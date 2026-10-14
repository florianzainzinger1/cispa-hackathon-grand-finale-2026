# Phase 1b findings — KGW signal

**Status:** VERIFIED and extracted for all three splits, merged into cache.

## Reconstruction

- By-the-book via `external/lm-watermarking` (`WatermarkDetector` internals) on **CUDA** (Quadro RTX 8000 on the JURECA login node — no Slurm needed; ~4,400 tok/s, test split in 5 min).
- Config as YAML: `ff-anchored_minhash_prf-4-True-1306382177`, γ=0.25, `select_green_tokens=true`, `ignore_repeated_ngrams=true`. Mechanics: token i is scored via the 4-gram **including itself** (self_salt) — PRF seeds a CUDA-Philox `randperm(V)`, green = first γ·V of the permutation.
- vocab_size **doesn't matter** between 151665/152064: CUDA randperm shares the random-key prefix stream across nearby sizes → identical greenlists for existing ids (verified: bit-identical sanity stats). Standardized on 152064.
- First `context_width−self_salt = 3` positions of each doc are unscored (`kgw_green = −1`).

## Sanity gate: PASSED (strong)

| | train | validation |
|---|---|---|
| green rate \| label=0 | 0.2525 (null 0.25) ✓ | 0.2422 ✓ |
| green rate \| label=1 | 0.2940 | 0.2999 |
| segments ≥50 tok, rate>0.35 | wm **67/291** vs clean **10/269** | wm **65/295** vs clean **9/260** |

≈23% of spans light up → KGW's span share ≈ ¼, as expected.

## Per-scheme metric (windowed binomial evidence, min-p over windows)

| split | variant | TPR@0.1%FPR | TPR@1%FPR |
|---|---|---|---|
| train | all scored positions | 0.0102 | 0.0532 |
| train | **first-occurrence-only** | **0.0159** | **0.0711** |
| validation | all scored | 0.0000 | 0.0389 |
| validation | **first-occurrence-only** | **0.0066** | **0.0461** |

- `ignore_repeated_ngrams` (first-occurrence-only) is uniformly better → use `kgw_first` as validity mask downstream.
- Absolute numbers at 0.1% FPR are still small: freak clean windows clog the tail (same story as Unigram). Expected lift from fusion (Phase 2–3) and tail cleaning (Phase 4/5). Per-scheme ceiling ≈ 0.25 (KGW's token share).

## Cache columns added (all splits)

`kgw_green` (int8: 1/0, −1 = unscored), `kgw_first` (int8: first occurrence of the 4-gram in doc). Extraction script: cluster `output/extract_kgw.py`; merge: [merge_signals.py](../src/merge_signals.py).
