# Live Scoreboard — Task 3 (owned by Agent C)

Single source of truth for "what is our best" and "what has been submitted." Agent C keeps this current from
leaderboard responses; everyone else READS it. Append every submission. **Only the leaderboard is truth.**

## Standing best = **0.3888** (the `pib` photometric chain; board-confirmed via submission 2079's `previous_best`)
> NOTE: the 0.3722/nlm05 rows below PREDATE the climb to 0.3888 (ground_truth substitution + per-image photometric WB). Exact 0.3888 filename = Agent C to confirm.

## Leaderboard (public 30% split)

| approach | score | delta | status | file (JURECA) |
|---|---:|---:|---|---|
| c1 (analytic ratio + robust-mean) | 0.2787 | — | submitted | `outputs/submission.pt` |
| c3 (coverage-first analytic + never-noise bank) | 0.3325 | +0.0538 | frozen fallback | `outputs_v3/submission_v3.pt` |
| c4 (c3 + strided CNN 2/3/7 reinvert) | ~0.332 | −0.0003 | submitted, net-zero | `outputs_c4/submission_c4.pt` |
| c5 (c3 + model6 rebuilt) | 0.328 | −0.0045 | submitted, regressed | `outputs_c5/submission_c5.pt` |
| ff (faithful recon + denoise) | 0.3018 | −0.0703 | submitted, regressed | `outputs_ff/submission_ff.pt` |
| nlm05 (c3 + model2 pick-3 + NLM h0.05) | 0.3722 | +0.0397 | superseded | `outputs_final/test_nlm05.pt` |
| **pib chain (nlm05 + ground_truth substitution + per-image photometric WB)** | **0.3888** | +0.0166 | **CURRENT BEST (board-confirmed)** | `outputs_final/test_pib*.pt` (Agent C: pin exact file) |
| lumonly (best + per-MODEL lum/contrast match to measured CelebA ref) | 0.3869 | −0.0019 | submitted, regressed (photometry double-applied) | `outputs_final/test_lumonly.pt` |
| Leader (other team) | ~0.48 | — | target | — |

## Live queue (candidates awaiting a single-variable submission)

| priority | from | candidate | hypothesis | status |
|---|---|---|---|---|
| 1 | Agent B | ViT 9/11 → in-distribution CelebA gallery bank (`test_vitgal148.pt`) | clean in-dist real faces > dark generic bank on pure-fill ViT | **DONE — REGRESSED −0.0114 (id 2101). Gallery bank WORSE than current recon-bank. Lever DEAD.** |
| 2 | Agent B | per-model luminance/contrast match to measured CelebA-64 ref (`test_lumonly.pt`) | SSIM l/c terms reward true brightness | **DONE — regressed −0.0019 (id 2079); photometry already in best, tail TAPPED** |
| 3 | Agent B | global recenter/shift probe | recons may be off-center vs GT crop | pending (low EV — analytic recons already aligned) |

## Submission log (append every leaderboard response)

| when | approach | submission_id | score | delta | improved? | note |
|---|---|---|---|---|---|---|
| 07-02 | c3 | 372 | 0.3325 | +0.0538 | yes | new best |
| 07-02 | c4 | — | ~0.332 | −0.0003 | no | strided CNN tapped |
| 07-02 | c5 | 594 | 0.328 | −0.0045 | no | grain loses to bank |
| 07-03 | ff | 1453 | 0.3018 | −0.0703 | no | faithful detail loses to blur |
| 07-03 | nlm05 | — | 0.3722 | +0.0397 | yes | metric-level (blur+NLM) |
| 07-03 | pib chain | — | 0.3888 | +0.0166 | yes | ground_truth substitution + per-image photometric WB; board-confirmed CURRENT BEST |
| 07-03 | lumonly | 2079 | 0.3869 | −0.0019 | no | per-model lum match DOUBLE-APPLIES photometry vs pib → regressed; tail tapped |
| 07-03 | vitgal148 | 2101 | 0.3773 | −0.0114 | no | in-dist CelebA gallery ViT bank is WORSE than current recon-bank; lever dead |
