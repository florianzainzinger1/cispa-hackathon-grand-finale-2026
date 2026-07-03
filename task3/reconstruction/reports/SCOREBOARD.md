# Live Scoreboard — Task 3 (owned by Agent B)

The single source of truth for "what is our best per model" and "what is the current
submission." Agent B keeps this current; everyone else READS it. Update on every build.

## Leaderboard (public 30% split)

| approach | score | delta | status | file (JURECA) |
|---|---:|---:|---|---|
| c1 (analytic ratio + robust-mean, noise-pad) | 0.2787 | — | submitted | `recon_work/outputs/submission.pt` |
| c3 (coverage-first analytic + never-noise bank) | **0.3325** | +0.0538 | **current best** | `recon_work/outputs_v3/submission_v3.pt` |
| c4 (c3 + strided CNN 2/3/7 re-inverted) | ~0.332 | −0.0003 | submitted, net-zero | `recon_work/outputs_c4/submission_c4.pt` |
| c5 (c3 + model6 rebuilt) | 0.328 | −0.0045 | submitted, regressed | `recon_work/outputs_c5/submission_c5.pt` |
| a1 (c3 + SVD-denoised MLP + analytic ViT) | ? | ? | BUILT, unscored | `recon_work/outputs_a1/submission_a1.pt` |
| **Leader (other team)** | **~0.48** | — | target | — |

## Per-model best source (what the current-best submission uses)

| model | family | source in current best | proxy score | candidate to beat it |
|---|---|---|---|---|
| 1 | MLP | analytic (clean) | — | retrieval / SVD-denoise |
| 2 | CNN s4 | low-res conv-invert (blurry) | — | retrieval / prior |
| 3 | CNN s8 | low-res conv-invert (weakest) | — | retrieval / prior |
| 4 | MLP | analytic (clean) | — | retrieval / SVD-denoise |
| 5 | MLP | analytic (clean) | — | retrieval / SVD-denoise |
| 6 | CNN s1 | ~46 conv-invert + 82 bank | — | trap-diff / BSS |
| 7 | CNN s4 | low-res conv-invert (blurry) | — | retrieval / prior |
| 8 | MLP | analytic (noisy) | — | SVD-denoise / BSS |
| 9 | ViT | diverse bank | — | retrieval / DPS prior |
| 10 | CNN s1 | conv-invert (trap-like) | — | trap-diff |
| 11 | ViT | diverse bank | — | retrieval / DPS prior |
| 12 | CNN s1 | conv-invert | — | trap-diff / BSS |

_proxy score = gradient self-consistency + subspace residual (Agent B's harness); GT is leaderboard-only._

## Submission log (append every human-reported delta)

| when | approach | submission_id | score | delta | improved? | note |
|---|---|---|---|---|---|---|
| — | c3 | 372 | 0.3325 | +0.0538 | yes | new best |
| — | c4 | — | ~0.332 | −0.0003 | no | strided CNN tapped out |
| — | c5 | 594 | 0.328 | −0.0045 | no | grain loses to clean bank |
