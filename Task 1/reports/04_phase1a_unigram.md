# Phase 1a findings — Unigram signal

**Status:** mask VERIFIED correct; signal real but weak (small span share × strength=1.0). Cache built for all splits.

## Mask reconstruction

- By-the-book reproduction of `external/Unigram-Watermark/gptwm.py` (pinned commit): `np.random.default_rng(sha256(int64(key)).digest()[:4] LE)`, `[True]*int(0.5*V)+[False]*rest`, shuffle.
- `vocab_size = model.config.vocab_size = 152064` (confirmed in repo's `run_generate.py`; also the only size covering max data token id 151909).
- **Verification:** 18 mask variants (vocab sizes × numpy/torch-CPU/torch-CUDA × sha/raw seeding) tested with global label contrast AND per-span exact binomial tests. Only the by-the-book mask lights up genuine spans: e.g. `validation_44@16` len=287 green-rate **0.80** (p=1.7e-25), 9 validation spans p<1e-4. Global averages were misleading (as the strategy review predicted): Unigram covers only a small share of spans (~10%), and strength=1.0 gives typical in-span green rates of only ~0.61.

## Signal quality (honest numbers)

| split | variant | TPR@0.1%FPR | TPR@1%FPR |
|---|---|---|---|
| train | raw | 0.0015 | 0.0212 |
| train | excl. repeated-2grams & specials | 0.0036 | 0.0181 |
| validation | raw | 0.0000 | 0.0261 |
| validation | excl. | 0.0016 | 0.0128 |

Per-scheme ceiling is ~its token share (~10%); Unigram alone is a minor contributor. Expected: KGW (γ=0.25, δ=1.5) and especially Gumbel/TextSeal (continuous per-token evidence) are the heavy lifters.

## Key insight: degenerate loops sit on BOTH sides

Repetitive text loops (e.g. `train_23` alternating special/newline pattern, `validation_37` 92-token all-green clean segment) produce extreme binomial evidence **in clean AND watermarked regions**:
- raw features → clean loops dominate the FP tail (fatal at 0.1% FPR),
- excluding repeated 2-grams → also kills genuinely watermarked degenerate spans (validation TPR@1% halves).

→ This is a **fusion-stage decision** (Phase 4): keep both raw + excluded features in the cache and let the tail calibration decide, possibly asymmetric (exclusion only for scoring clean-looking regions). Matches strategy §5.

## Cache

`cache/{train,validation,test}_signals.parquet` — per token: document_id, position, token_id, label (train/val), `unigram_green`, and degenerate columns `is_special_token`, `is_repeated_ngram_{2,3,4}`, `token_piece_length`, `is_number_like`, `is_newline_like`. KGW/Gumbel/TextSeal columns will be merged in by Phases 1b/1c.
