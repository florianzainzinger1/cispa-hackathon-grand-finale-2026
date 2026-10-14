# Phase 4 findings — HMM posterior

**Status:** the big lever worked. Public LB **0.0319 → 0.1568** (sub_003, rank ~3); tuned variant (val 0.2900) submitted as sub_004.

## Model ([hmm.py](../src/hmm.py))

- **5 states**: Clean + one W-state per scheme. No direct W→W′ transitions (spans are separated by ≥12 clean tokens per EDA).
- **Emissions factorize** over the four per-token signals (independent PRFs/keys): binned categorical distributions per signal — {0,1,missing} for unigram/kgw greens, 12 score bins for gumbel/textseal. Clean-state AND foreign W-states see the null distribution; only the own W-state sees the watermarked distribution. Invalid positions (v1-dedup repeats, unscored, special tokens for unigram) are missing in all states → contribute nothing.
- **Emission estimation**: label-1 spans assigned to a scheme by max mean window evidence (train span assignment: unigram 1,165 tokens, kgw 10,462, gumbel 10,656, textseal 11,371 — confirms unigram is rare, others ~equal). Clean bins from label-0 tokens. Add-0.5 smoothing.
- **Transitions**: geometric, hand-set then grid-tuned on validation with train-only emissions (no contamination): plateau at p_enter≈0.008–0.016, mean_span≈40–60 → **p_enter=0.012, mean_span=50**. Robustness cross-check (fit on val → eval train) passed.
- **Score** = −log10 P(Clean | all doc signals) via forward–backward, squashed ev/(ev+8).

## Why it beats windows

Emissions are strictly per-token → no boundary bleed by construction, AND span edges are recovered (windows sacrificed ~w/2 tokens per edge; the posterior localizes switches within a few tokens). Both Phase 2–3 problems solved simultaneously.

## Results

| variant | val @0.1%FPR | val @1%FPR | public LB |
|---|---|---|---|
| min(L,R) window fusion (Phase 2–3) | 0.0764 | 0.2241 | 0.0319 |
| HMM, train-fit emissions, default transitions | 0.2351 | 0.3983 | **0.1568** |
| HMM, tuned transitions (0.012/50), train+val emissions | **0.2900** | 0.4072 | **0.1901** |

Also tried, no gain (kept 12 bins / uniform entry): finer 18-bin emissions (val 0.2770 — estimation variance beats resolution), scheme-share-weighted entry probabilities (0.2819).

**FP-tail composition at 0.1% (validation, 52 clean tokens above threshold):** 41/52 lie within 10 tokens of a true span (posterior boundary smear, ~irreducible without sharper per-token info), 11/52 further away, concentrated in 5 documents. → Next information gain must come from per-token entropy (Phase 5), not from more transition/bin tuning.

Leaderboard at sub_003 time: #1 We Are All Alex 0.197, #2 SeeWeedZ 0.169, #3 us 0.157, #4 Polarnatt 0.155.

## Learned along the way

- Submission rate limit exists: ~180 s after a successful submission (HTTP 429 with wait time).
- Val→public discount ≈ 0.29→~? (sub_003: 0.235→0.157; pooled 0.1%-FPR operating points have high subset variance — applies to everyone).

## Next levers (Phase 5)

1. **Qwen entropy weighting** — background agent is computing per-token entropies for all splits; use as emission covariate (low-entropy tokens carry no watermark evidence → condition bins on entropy, or gate emissions).
2. Finer/adaptive score bins; per-scheme p_enter (unigram spans are rarer).
3. Learned fusion (LightGBM on posterior + window features, GroupKFold) only if it beats the HMM.
4. Duration modeling (negative-binomial via state chains) if transitions remain the bottleneck.
