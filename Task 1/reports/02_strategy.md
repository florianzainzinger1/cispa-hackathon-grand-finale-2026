# Strategy — Task 1: Text Watermark Localization

**Status:** agreed strategy (merged from Claude + ChatGPT analysis, 2026-07-02). Supersedes ad-hoc planning in chat.
**Companion doc:** [01_project_overview.md](01_project_overview.md) (facts, dataset, config).

---

## 0. Problem framing — what actually wins here

Per-token confidence in [0,1] that the token was generated **while a watermark process was active**. Evaluated as **TPR @ 0.1% FPR, pooled over all test tokens**.

Three consequences that drive every design decision:

1. **Only the global ranking matters.** Absolute score values are irrelevant; the metric is one point on the pooled ROC curve. "Score shaping" (0.99 vs 0.9) changes nothing unless it changes ranking.
2. **Scores must be comparable across documents** (pooling!). Per-document normalization that shifts distributions differently per doc would corrupt the pooled ranking. Our common currency: **p-values under the clean-text null hypothesis** — comparable across documents by construction.
3. **The battle is in the clean-score tail.** ~1.39M test tokens; if ~50% are clean (as in train/val), 0.1% FPR allows only **~700 false positives among ~700k clean tokens**. The decisive question is not "how good is the average score" but "which clean tokens produce the 700 most extreme null scores" — degenerate/low-entropy tokens are the main suspects and get special treatment (§5).

We are NOT building: a generic AI-text detector (labels ≠ "is AI text"; clean LLM text is class 0), a large trained model (signal is analytic, we hold the keys), or per-token isolated decisions (single tokens carry ~1 bit; only windows/spans are decisive).

**Core advantage:** we have the exact keys and parameters of all four schemes (see 01, §9). The task is signal reconstruction + localization, not blind detection.

---

## 1. Stage 1 — Per-token signal extraction (expensive, cached once)

For every token position i in every document (all splits), compute per scheme:

| Scheme | Per-token raw signal | Mechanics (params from `watermark_config.yaml`) | Compute |
|---|---|---|---|
| **Unigram** | `unigram_green[i]` ∈ {0,1} | ONE fixed global greenlist: key=1873092841, fraction=0.5. Membership lookup. | CPU, trivial, **build first** |
| **KGW** | `kgw_green[i]` ∈ {0,1} | Greenlist depends on previous tokens: seeding `ff-anchored_minhash_prf-4-True-1306382177`, gamma=0.25, `select_green_tokens=true`. Uses `torch.randperm` — **must run on CUDA (Philox)**, CPU gives wrong permutations (~⅓ signal loss). Also record `kgw_is_repeated_ngram[i]` (config: `ignore_repeated_ngrams=true`). | GPU Slurm job |
| **Gumbel-Max** | `gumbel_u[i]` ∈ (0,1) | PRF value of the observed token, seeded by (secret_key=2004683203, previous ngram=2). Score per token: `-log(1-u)` ~ Exp(1) under null. Via textseal repo (`provides: Gumbel-Max`), scoring_method v1. | GPU/CPU per repo |
| **TextSeal** | `textseal_score[i]` (+ per-key parts) | Dual key (key_a=947821031, key_b=1562881159), ngram=3, mixing_alpha=0.5, scoring v1. Repo entry points: `textseal.watermarking.config.WatermarkConfig`, `textseal.watermarking.detector.TextSealDetector`. TextSeal's *localized detection* is prior art for exactly our task — mine the repo for its windowing logic, not just raw scores. | GPU/CPU per repo |

Also per token, **from day one** (these columns explain the fatal clean-tail false positives later; recording them costs nothing now, retrofitting costs a re-run): `is_special_token[i]` (chat markers etc.), `is_repeated_ngram_{2,3,4}[i]`, `token_piece_length[i]`, `is_number_like[i]`, `is_newline_like[i]`, `token_id[i]`, position.

**Critical implementation rules:**
- Score the **given `token_ids`** directly (detectors in "detect" mode); never re-generate, never re-tokenize `text`.
- Tokenizer pinned: Qwen/Qwen2.5-7B-Instruct @ `a09a3545…` (downloaded to `data/tokenizer/`).
- Guard: `assert torch.cuda.is_available()` in the KGW extractor.
- Output: one Parquet per split (`cache/{split}_signals.parquet`), columns = document_id, position, all raw signals, label (where available). **Everything downstream runs on these files in seconds** — extraction runs once per split on the cluster.
- **Sanity gate per scheme** (ranking-based, NOT global label-1 averages — label=1 mixes all four schemes, so a correct Unigram detector can look weak when averaged over TextSeal spans):
  1. Compute the scheme's window evidence on train.
  2. Check the **top-k highest-evidence windows are strongly label=1 enriched**.
  3. Compute `tpr_at_fpr(labels, scheme_scores, 0.001)` **per scheme** — each scheme must beat the random baseline on its own before we build on it.
  If a scheme fails → our reconstruction is wrong (key/seeding/device bug); fix before building anything on top. Expected per-scheme ceiling is ~its share of spans (schemes don't detect each other's spans).

## 2. Stage 2 — Local evidence: windows → p-values

Single tokens are useless (a clean token is "green" with prob γ). Signal = local aggregation:

- **Window sizes:** 16, 32, 64, 128 tokens.
- **Window shapes:** centered, **left-only and right-only** (asymmetric). At a true span boundary the left/right statistics diverge sharply — this is our boundary detector and un-blurs the ~window/2 smearing that centered windows cause.
- **Per window, per scheme → exact p-value under the clean null:**
  - Green-list schemes: green count in window ~ Binomial(n, γ) under null → exact binomial tail p-value. Skip repeated-ngram positions for KGW (per config).
  - Gumbel: sum of `-log(1-u)` over window ~ Gamma(n, 1) under null → tail p-value.
  - TextSeal: use repo's v1 scoring statistic; if it doesn't expose a null distribution, calibrate empirically on train clean tokens.
- Work in `-log10(p)` for numerical comfort.

## 3. Stage 3 — Fusion and per-token score

- **Primary (analytic, robust): min-p fusion.** Per token: best (smallest) p-value across the 4 schemes × window configurations. Rationale: a span carries exactly ONE scheme — averaging dilutes a strong single-scheme signal; max-of-calibrated-evidence (= min-p) preserves it. Raw z-scores are NOT comparable across schemes; p-values are.
  - **Multiplicity correction note:** with the SAME number of tests k per token, Šidák (`1-(1-p)^k`) is monotone and **does not change the pooled ranking at all** — the metric is ranking-only, so the raw-vs-corrected debate is moot except where k varies (near document edges, fewer valid windows). We store both `min_p_raw` and `min_p_corrected`, use corrected where k varies, and let validation confirm.
- **Smoothing to spans:** labels are contiguous (≤6 spans/doc). Escalation ladder — each step must beat the previous on `tpr_at_fpr`, otherwise it is dropped:
  1. min-p **without** any smoothing (baseline to beat),
  2. simple peak/span filters (suppress isolated 1–2-token peaks, require minimum run length),
  3. **`ruptures` change-point detection** → candidate spans → span-level test → token scores,
  4. **HMM posterior as challenger** (2 or 5 states, emissions = per-scheme window evidence, transitions from train span statistics). Known risk at 0.1% FPR: an HMM can inflate one freak clean region into a full false span — exactly the failure the metric punishes hardest. Judged ONLY by pooled tpr_at_fpr on validation, never by visual span quality.
- **Optional (Stage 3b): learned fusion.** LightGBM / logistic regression on the ~20–40 cached window features. ~190k labeled tokens is plenty for this dimensionality, BUT the real diversity is only 180 documents → evaluate with **GroupKFold grouped by `document_id`** (neighboring tokens of one document must never straddle train/fold-val, else the estimate is optimistically biased). **Ship it only if it beats the best analytic variant on the local metric.** No deep learning: the signal is analytic, we hold the keys.

## 4. Stage 4 — Calibration for the metric

- Implement `tpr_at_fpr(y_true, scores, fpr=0.001)` **exactly as pooled** over all tokens; this single function is the local judge of every idea. Sanity-check it against sklearn ROC on synthetic data.
- Measure the **clean-tail**: which validation clean tokens rank highest? Inspect them manually (they tell us the failure modes — expected: repeated ngrams, boilerplate, special tokens, short gaps between spans).
- Final mapping to [0,1] is any monotone squashing of `-log10(p)` / posterior — ranking is what counts.
- Guard against validation overfitting: calibration choices (window set, HMM transitions) picked on train, verified once on validation; public LB (30%) used sparingly as the final check.

## 5. Stage 5 — Degenerate-token handling (the FP tail cleaner)

Low/zero-entropy tokens (forced continuations, repeated ngrams, chat markers, numbers/dates, code boilerplate) carry no watermark information but can accumulate freak statistics → they are the likely occupants of the fatal top-700 clean tail. Measures:
- KGW: honor `ignore_repeated_ngrams=true` in window counts.
- Exclude/downweight `is_special_token` positions in window statistics (they still get a score — required — but a conservative one derived from their span context).
- If needed later: Qwen2.5-7B forward pass → per-token entropy weighting (TextSeal's own trick, entropy-weighted detection). GPU job; only if validation tail analysis shows it pays.

## 6. Engineering & deployment

```
Task 1/
├── data/…                  # dataset, YAML, tokenizer (done)
├── external/…              # 3 detector repos @ pinned commits (done, local + cluster)
├── src/
│   ├── detectors/          # kgw.py, unigram.py, gumbelmax.py, textseal.py → signals
│   ├── features.py         # windows → p-values (reads cache, writes cache)
│   ├── fusion.py           # min-p, HMM, (lightgbm)
│   ├── metrics.py          # tpr_at_fpr (pooled) — THE judge
│   ├── predict.py          # cache → submission.jsonl
│   └── validate_submission.py  # reuse task_template.validate_predictions
├── cache/                  # {split}_signals.parquet, {split}_features.parquet
├── configs/                # yaml per experiment
└── submissions/            # sub_NNN_<hypothesis>.jsonl + submissions_log.md
```

- Heavy extraction on JURECA (GPU: sbatch, account `training2625`, reservation `cispahack`, partition `dc-gpu`); iteration on cached Parquet locally or on login node.
- Every submission = one named hypothesis, logged in `submissions/submissions_log.md` with local val score + public LB score. Leaderboard keeps team best; a worse experiment costs nothing but information.
- Submission endpoint: `POST http://<SUBMIT_SERVER>/submit/30-watermark-localization`, `X-API-Key` from `.env`.
- **Secret hygiene:** the API key lives ONLY in `.env` (gitignored). Never in reports, configs, code, logs, screenshots, or external chats (incl. ChatGPT). It has already been pasted into at least one external chat — if the organizers offer key rotation, rotate; either way, stop the bleeding now.

## 7. Phase plan (each phase ends submittable)

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | **Measuring instrument first:** `metrics.py` (pooled tpr_at_fpr, tested on synthetic data), `validate_submission.py`, `load_data.py` + **label EDA** (span lengths, spans/doc, gap lengths, span positions — needed for span filters/HMM priors later). Then dummy submission as endpoint test. | 1 h |
| 1a | Unigram signals (CPU) + Parquet cache schema (incl. degenerate-token columns) + per-scheme sanity gate + first local eval + top-FP inspection | 1–2 h |
| 1b | KGW signals via CUDA Slurm job (same sanity gate) | 1–2 h |
| 1c | Gumbel + TextSeal signals via textseal repo | 2–4 h (most reading) |
| 2–3 | Windows → p-values → min-p fusion → `sub_002_minp_raw`, `sub_003_minp_span_filtered` | 1–2 h after 1 |
| 4 | Smoothing ladder (filters → ruptures → HMM challenger) + tail calibration → `sub_004_smoothed` | 2–3 h |
| 5 | Learned fusion (GroupKFold) + entropy weighting if tail analysis demands → `sub_005+` | iterate |

Parallelization: 1a/1b/1c are independent — teammates can own 1c (textseal repo reading) while Claude does 0/1a/1b/2–3 scaffolding.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| KGW CPU/CUDA trap | CUDA assert + sanity gate (green rate on labeled spans) before trusting any KGW feature |
| Detector repo API mismatch vs YAML (e.g., scoring_method v1 semantics) | Sanity gate per scheme on train labels; pinned commits already match YAML |
| textseal repo complexity delays Gumbel too | Gumbel PRF is simple enough to reimplement standalone if repo fights us; validate against repo on samples |
| Val ≠ test distribution / public-private split variance | Analytic p-values (no heavy fitting), conservative choices; don't chase public LB noise |
| Unknown submission rate limits | Submit deliberately, log everything, keep best-known-good ready |
| "Other watermark schemes" as class-0 distractors | They should look ~null under our 4 keyed detectors (keys don't match) — exactly why keyed p-values beat generic detectors; verify on val clean tail |
| Shared cluster etiquette | Heavy jobs via Slurm only; caches inside team folder |

## 9. Decisions taken (with rationale, so we stop re-litigating)

1. **p-values as common score currency** — pooled metric demands cross-document comparability.
2. **min-p (max evidence) fusion, not averaging** — one scheme per span; averaging dilutes.
3. **Analytic-first, ML-second** — we hold the keys; a classifier only earns its place by beating min-p+HMM on `tpr_at_fpr` locally.
4. **Smoothing is an escalation ladder, not a favorite** — min-p bare → span filters → ruptures → HMM challenger; every rung must beat the previous on pooled tpr_at_fpr or it is dropped. (Revised after review: HMM's posterior elegance doesn't outweigh its false-span risk at 0.1% FPR — it earns its place empirically or not at all.)
5. **Asymmetric windows** — boundary sharpness without giving up window stability.
6. **Degenerate tokens are a first-class concern** — the 0.1% FPR budget lives or dies in the clean tail.
7. **No generic AI-text detectors, no SynthID/other schemes** — wrong label semantics / not in dataset.
8. **Sanity gates are ranking-based and per-scheme** — global label-1 averages mislead because label=1 mixes four schemes. (Revised after review.)
9. **Learned fusion is evaluated with GroupKFold by document_id** — token count is large but document diversity (180) is the true sample size. (Revised after review.)
10. **Metric + validator + label EDA before any real submission** — build the measuring instrument first; everything else is judged by it.
