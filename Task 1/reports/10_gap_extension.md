# Gap-extension round (after reaching rank 1 with 0.3362)

Goal: widen the lead. Current: val 0.4255 @0.1%FPR, public 0.3362 (#2 at 0.238).

## Per-scheme recall diagnosis at the 0.1% threshold (validation)

| scheme | wm tokens | recall |
|---|---|---|
| gumbel | 14,703 | 0.559 |
| textseal | 14,731 | 0.597 |
| **kgw** | **11,605** | **0.153** |
| unigram | 3,286 | 0.022 |

→ Gumbel/TextSeal carry us; KGW (26% of wm mass) and unigram are the untapped reserves.

## Tried in this round

1. **Argmax-match (Gumbel determinism)** — predict the token the watermark would have chosen (`argmax u^(1/p)` over full vocab, we hold keys + model) and test identity with the actual token. Signal exists and is strong (high-entropy: clean match 15% vs gumbel-span ~64%; overall in-span ~85% under T=1), **but redundant given u|logprob conditioning**: val 0.4266 vs 0.4255 baseline = noise. Split-half "gain" (+37%) was an artifact of coarser u-bins under a small fit set. NOT adopted. (Extraction script kept: cluster `output/extract_argmax_match.py`; train/val columns `gumbel_match, ts_match_a, ts_match_b` merged into caches.)
2. **KGW conditioning on logprob/both** — the gradient exists (in-span green rate 0.55 at lp<−3 → 0.28 at high lp; clean flat 0.25) but pooled val gets WORSE (0.400/0.410 vs 0.426): thin 2-bin likelihood ratios add clean-tail variance through the shared threshold. KGW with a binary γ=0.25/δ=1.5 signal is structurally capped; exact per-position green-mass is infeasible (self-salt seeding → one randperm per CANDIDATE token). KGW stays on entropy conditioning. CLOSED.
3. **Transition ensemble (5 plateau configs, mean/rank-avg)** — neutral (0.4243–0.4248 vs 0.4255). Not adopted.
4. **Unigram p_green — CLOSED (no gain)** — the exact computation works and matches theory beautifully (clean green rate tracks p_green exactly: 0.163 at pg<0.35, 0.858 at pg>0.65; wm rates match p_g·e/(p_g·e+1−p_g), e.g. observed 0.267 vs predicted 0.259). But pooled val: 0.4264 vs 0.4255 (noise), unigram recall unchanged (0.023). At strength=1.0/fraction=0.5 the per-token LR (~1.2–1.6) cannot accumulate past the 0.1% threshold within ≤320-token spans. **Unigram is fundamentally under-powered for this metric; its ~3.4% token share is unreachable.** Not submitted (tie-variance rule). Conditioning mode `pgreen` remains in hmm.py; train/val `unigram_pgreen` columns in caches.

## Second burst (after SarAI jumped to 0.3256, 0.011 behind us)

Eight further ideas tested against val baseline 0.4255, ALL ≤ noise:
doc-start prior fix p_start=0.29 (0.4240 — emissions dominate doc starts), max-u quantization spike bin (0.4228), 5 logprob levels (0.4239), TextSeal dual-key 2D emission u_A×u_B (0.4125 — bins too thin; fused+conditioning already carries it), Viterbi blend for boundary sharpening (0.4238–0.4244), eps smoothing sweep 0.1–32 (peak 0.4266 at eps=2, noise). Saturation conclusion is now backed by ~12 consecutive well-motivated negative results.

## Third burst: semi-supervised EM on test (the last big untried axis) — CLOSED

Idea: 1.39M unlabeled test tokens as additional emission-fit data (E-step posterior on test → M-step refit; honest check = val labels, never in the fit). Results: soft-count EM val 0.4176 (diffuse posteriors smear the distributions); hard pseudo-labels at conf>0.9/0.95/0.99 val 0.3777/0.3632/0.4016 — ALL clearly worse than 0.4255. Cause: confident pseudo-labels sample only span CORES → selection-biased wm distributions overweight extreme evidence. Self-training on this problem structure degrades; labeled fit stands.

## Fourth burst: four structured challengers (external review's proposal, champion frozen)

| Challenger | Result | Verdict |
|---|---|---|
| A: Hard-negative tail-veto (subagent; HGB on critical zone only, GroupKFold + cross-split + split-half) | best cell +0.0008 (vanishes under stricter eval); split-half flat-to-negative; only ~250 clean zone tokens/split, max P(clean)=0.26 | **NO-SHIP.** Root cause: the FP tail is genuinely ambiguous text, not learnable junk — features carry the same info the emissions already condition on. |
| B: Constrained-interval DP (≤6 intervals, ≥24 len, ≥12 gap, exact DP over posterior) | best +0.0019 at val-TUNED tau (protocol violation); others ±0.000 | **NO-SHIP** (noise once tau tuning is discounted; hard constraints add nothing over the soft posterior). |
| C: Per-document scheme suppression (two-pass, soft doc-level scheme prior) | 0.4194 vs 0.4255 at all floors | **NO-SHIP** (suppresses true rare-scheme spans more than FPs). |
| D-proxy: bigger reference model (14B/32B) | not run | REJECTED a priori: YAML pins Qwen2.5-7B-Instruct — the generator IS the 7B; a larger model is a worse proxy. |
| **D-rank: rank-conditioned Gumbel likelihoods** | val 0.4287 (+0.003), split A→B +0.020, B→A +0.031, val→train +0.025 — **all four directions positive** | **SHIP CANDIDATE** (sub_007 generated & validated, awaiting explicit approval). Rank is calibration-invariant — transfers across splits better than logprob. textseal stays logprob (rank hurts it: 0.4096). |

## Fifth burst: generation-temperature hypothesis (deep-research lever #1) — CLOSED

Hypothesis: watermark spans generated at T≠1 (textseal default 0.9), so our logprob-conditioning (computed at T=1) is miscalibrated. Recomputed per-token actual-token logprob at T∈{0.6,…,1.3} (one Qwen forward pass/doc, GPU), re-ran HMM conditioning. Result: val 0.410–0.429, non-monotone, no peak at 0.9, T=1.0 ≡ baseline 0.4257. **Refuted.** Reason: logprob is a *conditioning* variable (monotone reparam within a position → same information at any T); the Gumbel score −log(1−u) is temperature-independent by construction. Temperature was tested at the only place it could matter. Deep research (questions 2–4) still running.

## Round conclusion

All four identified reserves are now closed (KGW: structurally capped + infeasible exact computation; unigram: under-powered scheme; argmax-match: redundant; ensembles: neutral). **The current model (sub_006) sits at the plateau of what these signals + this metric allow.** Remaining theoretical headroom is in gumbel/textseal boundary tokens (recall 0.56/0.60) — no concrete idea beats the posterior there. Strategy shifts to: hold rank 1 (~0.09 lead), monitor competitors, prepare presentation.

## Ops notes

- Upload trap discovered: piping a file through `wsl ssh 'cat > remote'` combined with other commands in one call can silently produce a 0-byte file → empty python script exits 0 and `&&` chains still print success markers. ALWAYS `wc -c` remote scripts after upload.
- Leaderboard monitor runs (5-min polling, top 6, persistent).
