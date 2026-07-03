# Agent MX — model12 color-cast + model10 cleanup + strided-CNN SSIM-inversion

**Base:** `outputs_final/test_vithi45.pt` (team best 0.3837). Every candidate = base with EXACTLY
ONE model swapped (other 11 byte-identical — verified by `mx_verify.py` → ALL_SINGLE_VAR_OK).
All candidates VALID (12 models, 128 distinct, [0,1], finite). **Not submitted** (Agent C serializes).

CelebA reference (from `gallery/celeba_top200_cc178.pt` /255): gmean 0.4376,
chan_mean [0.515, 0.423, 0.375] (warm, R>G>B), chan_std [0.306, 0.284, 0.281].

## Candidates (ranked by likelihood of beating vithi45's version on SSIM)

### model12 (color-cast correction) — HIGH confidence
- **`outputs_mx/cand_m12_color1.pt`** — RANK 1 (submit first). Gray-world white-balance (per-channel
  additive shift to CelebA means) + mild per-image brightness lift (strength 0.5). NO contrast change,
  NO denoise (base already smooth), NO sharpening. gmean 0.408→0.437, chan_mean now exactly matches
  CelebA. Visual: same faces as base but brighter/warmer/natural skin, backgrounds lifted from black,
  ZERO added grain. Textbook low-freq fix.
- **`outputs_mx/cand_m12_color2.pt`** — RANK 2 (try if color1 confirms +). mean+std match (strength 0.7,
  std cap 1.25) + σ=0.5 low-pass. Slightly warmer/softer than color1. Clean, marginally more aggressive.
- **`outputs_mx/cand_m12_color3.pt`** — REJECT. Built from the raw trap recon (sharper) with heavy
  desaturate+NLM+low-pass, but still carries the sigmoid oversaturation/dark-bg/speckle. Grain = SSIM
  killer. Do not submit.

### model10 (dark + rainbow speckle) — HIGH confidence
- **`outputs_mx/cand_m10_clean1.pt`** — RANK 1. vithi45 base: despeckle (median-on-outliers removes the
  rainbow dots, leaves structure), brightness lift 0.276→0.40, mild WB/contrast toward CelebA, σ=0.4
  low-pass. Visual: SAME faces as base, speckle gone, correctly lit, no grain added. Best low-freq fix.
- **`outputs_mx/cand_m10_clean2.pt`** — RANK 2 (higher variance). Trap real faces: despeckle + brightness
  match + NLM + low-pass. Sharper/more detail but retains more trap grain/color-cast on ~1/4 of faces.
  Raw trap swap already tested NEUTRAL on nlm05 base; this is the de-grained version — try only if
  clean1 confirms + and we want more.

### strided CNN 2/3/7 (SSIM-optimal conv-inversion) — LOW confidence / likely regressions
GPU job 15402804 (COMPLETED). `invert_conv` with strong TV (2e-2 / 1.5e-2) on a low-res latent
(base_res = feature res) + Gaussian low-pass + CelebA color-match.
- **`cand_m2_ssiminv.pt`** — REJECT. Clearly worse than vithi45's model2 (the confirmed pick-3 +0.0088
  win): grainy, heavy color speckle, broken tiles. Would regress a known win.
- **`cand_m7_ssiminv.pt`** — REJECT/last-resort. More face structure than m3 but grainier than vithi45's
  current model7. Lateral-to-negative on grain.
- **`cand_m3_ssiminv.pt`** — MARGINAL only. model3 is stride-8 (8×8 feats) = signal-starved; both this and
  vithi45's model3 are bad. Color is corrected (0.478→0.436 warm) but structure stays blobby+speckly.
  The ONLY strided candidate where the baseline is equally bad, so a color-only edge is *possible* but
  low-EV. Submit only after model12/model10 wins are locked and if a cooldown is free.

## Verdict
Two genuine low-freq wins found: **model12 color1** and **model10 clean1** (both fix
darkness/cast/speckle with zero added grain — exactly what SSIM rewards). Strided-CNN inversion
reconfirmed TAPPED (matches the DEAD ledger); do not spend cooldowns there beyond the marginal m3.

## Recommended submission queue for Agent C
1. `cand_m12_color1.pt`  (model12 gray-world WB)
2. `cand_m10_clean1.pt`  (model10 despeckle + brighten)
3. `cand_m12_color2.pt`  (only if color1 +)
4. `cand_m10_clean2.pt`  (only if clean1 +, higher variance)
5. `cand_m3_ssiminv.pt`  (marginal, optional)
