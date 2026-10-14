# Phase 5 findings — entropy/logprob-conditioned emissions → RANK 1

> **UPDATE (sub_006, public 0.3362):** conditioning Gumbel/TextSeal emissions on token **logprob** instead of entropy is even stronger (val 0.3762 → **0.4255**). Physics: under Gumbel-max sampling the chosen token is argmax u^(1/p) — a low-probability token can only win with an extreme PRF value u, so wm-gumbel mean score is 1.74 for lp<−3 vs 1.10 for lp>−0.3, while clean stays flat (1.01). Same per-level normalization trick as entropy (that's why THIS logprob use transfers while the raw logprob channel failed). Best combo: unigram/kgw=entropy, gumbel/textseal=logprob (`--cond best`). Joint (both) conditioning is NOT better — bins get too thin.

**Status:** `sub_005_hmm_entropy.jsonl` public LB **0.2831** → **leaderboard rank 1** (Polarnatt 0.238, Loki 0.200, We Are All Alex 0.197).

## What worked: entropy conditioning (val 0.290 → 0.376)

- Qwen2.5-7B per-token predictive entropies for all splits (background agent, cluster GPU, ~3.1k tok/s combined entropy+logprob pass).
- **Verified premise:** clean PRF signals are entropy-independent (uniform by construction: gumbel mean 1.01 across all entropy levels), watermark signals grow with entropy (gumbel in wm spans: 1.12 / 1.22 / 1.48 across levels; kgw green rate 0.27→0.33).
- Implementation ([hmm.py](../src/hmm.py)): observation codes in product space signal_bin × entropy_level (4 levels, edges 0.3/1.0/2.0), emission distributions **normalized within each entropy level** so the text's own entropy doesn't leak into the likelihood ratio — only the signal-given-entropy does.
- Honest split-half pre-test (+34%), then fit=train → val: **0.3662**, transitions re-tuned (plateau, chose 0.02/45): **0.3762**.

## What failed (documented so nobody retries)

| idea | result | why |
|---|---|---|
| Logprob emission channel (11 bins, per-state dists) | val 0.376 → **0.005** | style detector, does not transfer across documents; swamps PRF evidence |
| Erlang-k duration states (k=2,3) | 0.290 → 0.249–0.268 | geometric flexibility helps the 0.1% operating point |
| finer entropy bins (5–6 levels) | 0.341–0.351 | estimation variance |
| finer score bins (18) | 0.277 | same |
| scheme-share-weighted entry probs | 0.282 | emissions dominate |
| empirical tail calibration (Phase 2–3) | val 0.000 | can't resolve p<2e-5 from 47k clean tokens |

## Val→public tracking

| sub | val (train-fit) | public | factor |
|---|---|---|---|
| 003 | 0.2351 | 0.1568 | 0.67 |
| 004 | 0.2900 | 0.1901 | 0.66 |
| 005 | 0.3762 | 0.2831 | 0.75 |

## Stacker resolution (Phase 5 learned-fusion: CLOSED)

Leak-free protocol (HMM posterior cross-fitted over 5 doc-folds, GBM via GroupKFold on those OOF features, pooled OOF metric):
- cross-fitted HMM baseline: **0.3212**
- HistGradientBoosting stacker (20 features incl. posterior, window evidence, entropy/logprob/rank, degenerate flags): **0.2759**
- rank-average blend: 0.2989

**The analytic HMM beats the learned model** — strategy §9.3 confirmed empirically; no stacker submission. Also tested: one EM round of posterior-based span re-assignment for emission fitting (0.4279 vs 0.4255 — noise, not adopted).

## Remaining ideas (if competitors catch up)

1. LightGBM stacker on posterior + window evidence + degenerate flags (GroupKFold by document_id) — logprob features only with per-document normalization, given the transfer failure above.
2. qwen_rank as a gated feature (rank==1 runs at low entropy = forced text, downweight).
3. Boundary micro-refinement: 2-pass HMM with position-dependent entry probs near detected span starts.
4. Unigram span-level rescue (its 0.003 per-scheme TPR leaves its ~3% token share mostly unclaimed).
