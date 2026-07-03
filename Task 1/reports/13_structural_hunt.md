# Report 13 — Structural-lever hunt (Jul 2 eve, post-v2)

Goal: find a *structural* lever (new information source), not tuning, to close the
0.094 gap to SeeWeedZ (0.4688) from our 0.3744 (#3). Scripts: `src/diag_recovery.py`,
`src/kgw_ceiling.py`, `src/v2_sweep.py`. All numbers = clean fit-train / eval-val, TPR@0.1%FPR.

## 1. Where recovery fails (diag_recovery.py)

| Scheme | % of WM tokens | recall @0.1%FPR | contribution to 0.4255 |
|---|---|---|---|
| gumbel | 31% | 0.589 | 0.183 |
| textseal | 32% | 0.615 | 0.197 |
| kgw | 24% | **0.175** | 0.041 |
| unigram | 13% | **0.039** | 0.005 |

KGW+unigram = 37% of WM tokens, barely recovered. That's where any lever must come from.

## 2. Is the signal absent or under-extracted? (raw separation)

- KGW green rate on WM spans **0.408 vs 0.249** clean (0.482 vs 0.249 at high entropy). Signal is REAL and strong.
- Unigram green 0.648 vs 0.507. Present, weaker.

So the signal is present. The question is why we don't recover it.

## 3. KGW ceiling (kgw_ceiling.py) — the global-threshold penalty

- In the **full model**, KGW recall = 0.175 regardless of conditioning (entropy/logprob/rank/both all ≈0.17); richer conditioning HURTS overall (shifts the global threshold, costs gumbel/ts recovery).
- A **KGW-only detector** (all other emissions muted) caps at **0.30** recall (logprob-conditioned).
- ⇒ KGW's ceiling isn't a detector bug: at a *global* 0.1% FPR, the threshold is set by the strong gumbel/textseal false-positive tail; weaker KGW spans fall below it. The exact per-token KGW LR (needs per-position greenlist base-mass `kgw_pgreen`, cluster-only) would lift KGW maybe 0.175→~0.22 in the joint model ⇒ **overall +0.02–0.04 at best**.

## 4. Low-entropy in-span tokens — propagation works, spans are the limit

Low-ent (<0.3) WM tokens are 31% of WM, recall 0.279. Split by their span's mean score:

| span strength | recall |
|---|---|
| weakest (0.03–0.65) | 0.003 |
| 0.65–1.46 | 0.357 |
| 1.46–1.84 | 0.592 |
| strongest (1.84–2.84) | 0.887 |

HMM propagation to deterministic tokens works (0.887 in strong spans). Tokens are lost only when the WHOLE span is weak — and spans are weak precisely in low-entropy text, which cannot carry watermark evidence. **Information-theoretic floor, not a bug.**

## 5. Verdict

Independent diagnosis CONFIRMS the chronicle's earlier conclusion: we are at the
information ceiling of per-token watermark detection.
- Strong schemes (gumbel/textseal, 63%) are well-extracted (~0.6–0.68 at usable entropy).
- Weak schemes (KGW/unigram, 37%) are threshold-capped; best realistic lever = exact KGW LR = **+0.02–0.04**, needs a cluster extraction, and does **not** change our rank (#2 is +0.06, #1 is +0.09 away).
- The 0.094 gap to SeeWeedZ is a DIFFERENT information source / method, not a detector refinement — consistent with the Discord-coordinated multi-team jump.

Recommendation: the only concrete detection lever left (exact KGW LR) can't change our rank. Higher-EV use of remaining time = Task 3 (deadline Jul 3 10:00, analytic leaks = near-free SSIM). Task 1 entry (0.3744, clean logprob champion) is private-safe as-is.
