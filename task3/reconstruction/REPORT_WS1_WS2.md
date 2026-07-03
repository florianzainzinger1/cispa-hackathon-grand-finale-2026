# WS1 (archetypal selection) + WS2 (oracle pin) — results

Base synced to **test_pib1.pt (0.3878)**. Read-only w.r.t. existing pipeline; only NEW
files added. No leaderboard submissions. All SSIM impact is **UNVERIFIED — needs your A/B.**
Deconfliction respected: only leaky-MLP {1,4,5,8} touched; models 2/7, ViT 9/11, and all
blend/postproc/mx_/merge_ scripts untouched.

## Files added (code/)
`archetype_select.py`, `ssim_grader.py`, `ssim_validate.py`, `arch_synth_eval.py`.
Oracle fix is a **drop-in** (`ssim_grader.py`); `synth_lib.py` / `gp_vit_prior.py` NOT modified.

## Artifacts (outputs_final/, candidates_arch/, reports/arch/)
- `outputs_final/base_archetypal.pt` — raw SPA vertices, {model1,4,5,8}.
- `outputs_final/base_archetypal_denoised.pt` — SPA vertices + within-identity averaging.
- `candidates_arch/<m>/grid_{archetypal,denoised,oldquality}.png` — visual side-by-sides.
- `reports/arch/{purity_report,purity_report_denoised,synth_spa_vs_old,ssim_convention_validation}.json`.

---

## WS1 — Extreme-point (SPA) selection

**Method.** Ratio rows r_i=gW[i]/gb[i] are convex combos of the batch; single-sample neurons are
the VERTICES (clean faces), interior points are blend-ghosts. We project rows into the rank-128
right-singular subspace of gW and run SPA (greedy max-norm + deflation) to pick the 128 purest
vertices, emitting them at exact [0,1] scale (`.clamp`, never per-image-minmax). Leaky set confirmed
{1,4,5,8} (net.0 Linear(1024,12288); acts sigmoid/tanh/relu/relu; ratio cancels activation → clamp
correct for all).

**Vertex-purity (convex-hull residual; higher = purer vertex, GT-free):**
| model | SPA mean | OLD-quality mean | Δ |
|---|---|---|---|
| 1 | 0.719 | 0.650 | **+0.070** |
| 4 | 0.740 | 0.589 | **+0.151** |
| 5 | 0.703 | 0.619 | **+0.084** |
| 8 | 0.756 | 0.623 | **+0.134** |

SPA uniformly picks purer vertices — hypothesis confirmed geometrically.

**Synthetic-oracle SSIM (true known-CelebA GT, per-channel):** on CLEAN gradients SPA beats the old
selection by **+0.087 (m1), +0.116 (m4), +0.043 (m5), +0.000 (m8)**.

### ⚠ THE KEY FINDING (honest, changes the recommendation)
The **real** leaky-MLP gradients are NOT clean: **E128 = 0.77 / 0.77 / 0.78 / 0.47** for m1/4/5/8
(~23% off-subspace noise). In that regime a single-sample vertex carries that face's FULL noise →
**raw SPA is grainy** (TV≈0.62), visibly grainier than the old quality selection (which deliberately
picks low-TV rows). Per your empirical law #2 (smooth beats faithful, grain kills SSIM) and law #1
(offline proxies don't track SSIM), **raw `base_archetypal.pt` will very likely REGRESS on the board
despite winning on purity.** The synthetic proxy at matched noise collapses both methods to ~0.02,
confirming the offline number can't vouch for raw SPA. **Do not A/B raw SPA expecting a gain.**

### The fix — `base_archetypal_denoised.pt`
SPA gives correct, diverse IDENTITIES; they just need de-graining. We average each vertex with its
≤5 nearest same-face siblings in the subspace (√k denoise; a vertex's neighbours are its own noisy
copies, so grain cancels without blending distinct identities). **TV halved** (m4 0.643→0.311,
m5 0.558→0.234); grids show clean, diverse faces. This is the candidate worth A/B-ing — it combines
SPA's identity coverage with the smoothness the board rewards. (Caveat: model4 denoised has 127
unique rows — one averaging collision; your build's uniqueness guard handles it, or re-run with a
dedup.)

**Recommended A/B (one variable, per your rule):** swap ONLY the 4 leaky-MLP blocks of `test_pib1.pt`
with `base_archetypal_denoised.pt` and submit. If it helps, iterate the denoise strength (k=3/5/8);
if not, raw SPA is confirmed dead and the current selection stands.

---

## WS2 — Pin the SSIM oracle convention

**Root cause (confirmed).** The oracle's `hungarian_ssim` scores in **channel-mean GRAYSCALE**
(`A.mean(1)`) — blind to per-channel colour/brightness — and `synth_lib`'s fast variant also matches
pairs by pixel-MSE, not SSIM. The grader (per the Task-3 PDF) matches one-to-one BY SSIM on RGB. The
grader's exact window is not shipped (server-side; venv has only skimage) → inferred empirically.

**Fix (`ssim_grader.py`).** Per-channel skimage SSIM (`channel_axis=0`, `data_range=1.0`), Hungarian
assignment computed ON the SSIM cost matrix, window selectable (default 7×7 uniform = skimage default;
11×11 gaussian σ=1.5 toggle), no shift alignment (grader does none).

**Validation (`ssim_validate.py`, vs replay-certified `ground_truth.pt`).** GT blocks are keyed by
source descriptor, not challenge index → matched by replay-cos. 5 models certified (m1↔0.98, m4↔0.99,
m5↔0.97, m10↔1.00, m12↔1.00). Re-scoring the three known submissions:
| convention | pib1 | lowsig | nlm05 | #1 pick |
|---|---|---|---|---|
| grayscale-7 | 0.4416 | 0.4386 | 0.4443 | **nlm05** (WRONG) |
| **per-channel-7** | 0.4226 | 0.4194 | 0.4213 | **pib1** (matches board #1) |

**Verdict.** Switching to per-channel **fixes the real mis-ranking**: it moves the true board leader
`pib1` from #2 (grayscale) to #1, exactly the colour-blindness the grayscale core caused. Neither
convention reproduces the *full* order and absolutes run high (~0.42 vs board 0.38) because the oracle
only covers **5 of 12** models (replay-certified; ViT/weak-CNN absent). 

**Answer to "can we trust offline testing?":** Partially. With the per-channel fix the oracle is
trustworthy for **coarse / colour** decisions (it now agrees with the board on the #1 candidate), but
NOT for **fine grain distinctions** (lowsig vs nlm05) — its GT coverage is incomplete, so it can't
substitute for a real A/B on close calls. Use it to REJECT clearly-bad candidates and to catch
colour regressions; keep the leaderboard as ground truth for close ones (your law #1 stands).
