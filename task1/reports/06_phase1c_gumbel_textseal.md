# Phase 1c findings — Gumbel-Max & TextSeal signals

**Status:** VERIFIED (numpy port bit-matched vs repo), extracted for all splits, merged into cache.

## Reconstruction

- The textseal PRF is **pure integer arithmetic** (prime-weighted context sum → mixing hash → mod 8191) — deterministic and device-independent; **no CUDA dependency** (unlike KGW). Ported to vectorized numpy: [gumbel_textseal.py](../src/detectors/gumbel_textseal.py).
- **Port validated against the repo** (`score_listed_tokens`, pinned commit) on cluster: max |diff| ≈ 4.8e-7 (float32 rounding) over 3 documents, PASS.
- Repo package import pulls `nltk` (absent in shared venv) — worked around by loading `config.py`/`core.py` via importlib without `textseal/__init__.py`.
- Mechanics per token at position ≥ ngram+1 (context = ngram tokens BEFORE the target):
  - **Gumbel-Max** (key 2004683203, ngram=2): u = PRF(ctx, tok, key), score = −log(1−u) ~ Exp(1) under null. Window sum ~ Gamma(n,1) → exact tail p-value.
  - **TextSeal** (key_a 947821031, key_b 1562881159, ngram=3, α=0.5): fused = α·(−log(1−u_a)) + (1−α)·(−log(1−u_b)); unit mean, variance α²+(1−α)²=0.5 → moment-matched Gamma tail (as in repo detector).
  - `scoring_method: v1` = dedup by context window → `*_first` flags (first occurrence of ctx in doc).
- Note: PRF modulus is only 8191 → u is quantized to 13 bits, max per-token score ≈ 9.01. Entropy weighting (repo's `TextSealDetector`) deferred to Phase 5 (needs Qwen forward pass).

## Sanity gates: PASSED

| split | signal | mean \| label=0 (null 1.0) | mean \| label=1 |
|---|---|---|---|
| train | gumbel_score | 1.0129 | 1.2698 |
| train | textseal_score | 1.0089 | 1.1171 |
| validation | gumbel_score | 1.0131 | 1.2296 |
| validation | textseal_score | 1.0223 | 1.1389 |

(label=1 mixes all 4 schemes → per-scheme in-span elevation is ~4× the diluted numbers.)

| split | scheme (v1-dedup, min-p over windows) | TPR@0.1%FPR | TPR@1%FPR |
|---|---|---|---|
| train | gumbel | 0.0111 | 0.0602 |
| train | textseal | 0.0172 | 0.0809 |
| validation | gumbel | 0.0105 | 0.0594 |
| validation | textseal | 0.0158 | 0.0761 |

## Cache columns added (all splits)

`gumbel_score` (float32, NaN = unscored first ngram+1 positions), `gumbel_first`, `textseal_score`, `textseal_first`.

**All four scheme signals are now verified and cached** → ready for Phase 2–3 fusion.
