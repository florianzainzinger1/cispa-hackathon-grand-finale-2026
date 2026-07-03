# Task 3 — FL Audit: Gradient-Inversion Approach

Leaderboard `TASK_ID = 21-fl-audit`. Reconstruct **1536 private training images**
(12 models × 128 images each) from **batch-128-averaged** federated-learning
gradients.

- **Images:** 64×64 RGB, CelebA-style faces, `float32` in `[0,1]`.
- **Submission:** one `.pt` → `dict{model1..model12: Tensor(128,3,64,64) float32 [0,1]}`, ≤300 MB.
- **Score:** per-model **Hungarian one-to-one mean SSIM** (matches each recon to its
  best GT partner), averaged over models. 30% public / 70% private split.
- **Rate limit:** 1 submission / 5 min. `POST http://<SUBMIT_SERVER>/submit/21-fl-audit`,
  header `X-API-Key` (from repo-root `.env`, gitignored — never commit it).
- **Account is SHARED with teammates — coordinate before every submission.**

Reference scores: my analytic+postproc track peaked at **#958 = 0.3035 public**;
teammate's metric-tuning pipeline reached **0.3828**; leaderboard leader ~0.48.

---

## 1. Core principle: analytic first-layer readout, per family

For any fully-connected layer `y = Wx + b`, the per-sample gradients satisfy
`∂L/∂W_i = (∂L/∂z_i)·xᵀ` and `∂L/∂b_i = ∂L/∂z_i`, so

```
    x  =  gradW_i / gradB_i          (Boenisch et al. Eq.6 / Geiping Prop. 3.1)
```

recovers the **input to that FC layer** exactly, for any neuron `i` whose bias
gradient is non-zero. This is the backbone of every non-ViT model here. The catch
under **batch-128 averaging** is that the readout returns a *superposition* of the
128 inputs weighted per neuron; we `dedup()` the recovered features by similarity
and keep the cleanest 128 (one dominant activation per neuron).

Everything downstream is: (a) undoing whatever sits *between* the readout point and
the image (conv/pool), and (b) **metric-level cleanup for SSIM** (§4).

---

## 2. Per-model attack map

| Models | Architecture | Readout point | Method | State |
|--------|--------------|---------------|--------|-------|
| **1, 4, 5, 8** | MLP | `net.0` FC input = the image | analytic readout → per-image face | **defended / noisy** — additive HF speckle grain |
| **10, 12** | CNN, full-res | `fc1` → conv inversion | analytic feature map, then invert conv (over-determined, clean) | clean structure, **neon chroma** in conv nullspace |
| **2, 3, 7** | CNN, **pooled** | `fc1` → *pooled* map → conv/pool inversion | analytic pooled map, then optimize image s.t. `pool(relu(conv(x)))` matches | **info-bottleneck** (see §3) |
| **6** | defended | — | light TV, keep gray | real DP-style noise, little signal |
| **9, 11** | ViT | — | single prototype fill (no clean per-image readout) | effectively dead; Geiping cosine-matching → mush at batch 128 |

Files: `mlp_attack.py` (1,4,5,8), `cnn_fullres_attack.py` (10,12),
`cnn_pooled_attack.py` (2,3,7), `geiping_attack.py` (ViT 9,11, cluster GPU job).

---

## 3. The pooled-CNN info-bottleneck (models 2, 3, 7)

Pipeline: `conv(3→8,3×3,pad1) → act → avg_pool(k) → flatten → fc1 → act → head`,
with `k = 64 / feat_H`.

The `fc1` readout sees the **pooled** feature map, so the analytic recovery is
**under-determined** relative to the image:

| Model | feat shape | pool k | measurements | vs 12288-px image |
|-------|-----------|--------|--------------|-------------------|
| 7 | (8,16,16) | 4 | 2048 | 0.17× |
| 2 | (8,16,16) | 4 | 2048 | 0.17× (**but analytic readout degenerate — `clean=0`**) |
| 3 | (8,8,8)   | 8 | 512  | 0.04× (worst) |

Within-pool-block high-frequency detail is **physically absent** from the
measurement — an information limit, not a method gap. The recoverable content is
**low-frequency**, so the correct move is to *seed it low-frequency and smooth it*.

### The inverter fix (`invert()` in `cnn_pooled_attack.py`)

Two changes vs the old gray-init / weak-TV baseline:

1. **Upsample init** — seed `x` with a per-image bilinear upsample of the pooled
   channel-mean (a luminance guess), min-max normalized into a mild `[0.2,0.8]`
   band, replicated to RGB. Starts optimization already in the right low-freq basin.
2. **Strong TV** (`tv=8e-3`, up from `1e-3`) over `iters=1500` Adam — smooths away
   the **pooling-grid block artifacts** at the source.

**GT-grounded round-trip validation** (invert from clean features, score against the
seed image):

```
model7 (k=4):  0.489 → 0.609   (+0.12 SSIM)   [tv 1e-3→1e-2 sweep, peak 0.609]
model3 (k=8):  0.303 → 0.356   (+0.05 SSIM)
```

model2's readout is degenerate (`clean=0` clean analytic features) so its inverter
runs on noisy features and the benefit is **unverified** — treat 2 as best-effort.

### Why this is *aligned* with the metric-track winning principle, not opposed

The metric track's hard-won lessons include:

- **#3 (winning):** *grain is the SSIM killer; smooth beats faithful.*
- **#12 (regressed −0.07):** *chasing "faithful" high-frequency detail hurts SSIM.*

The pooling-grid blocks are **grain**, not real detail — removing them at the source
is a direct application of #3, and is categorically different from #12's failed
attempt to *add* faithful HF detail. So the fix is a **base-recon improvement the
metric track never had underneath it**, and it is **composable**: the metric track's
blur / NLM / mean-blend / shrink all still run on top, now on a cleaner base.

---

## 4. Metric-level postprocessing (`postproc.py`) — where the gains live

**SSIM is low-pass biased.** Every leaderboard gain came from metric-level cleanup,
not from better gradient algebra. Per-model knobs in `CFG`:

- `method` — `nlm` (adaptive non-local-means, `h = den·σ̂`, so clean recons are barely
  touched and noisy blends are denoised hard) / `tv` / `bilat` / `none`.
- `sat` — chroma keep factor; kills the CNN neon-chroma nullspace (desaturate).
- `blur`, `med` — gaussian σ / median filter; smooth pooling seams & grid.
- `con` — **contrast-restore**: denoise strips speckle *and* flattens genuine face
  contrast (drops the SSIM contrast term); rescale per-image luminance-std back toward
  a target, only-restore-never-flatten, mean preserved.
- `shrink` — **confidence-weighted mean-face shrink**: confidence = `1 − normalized
  grain` read from the **raw** recon; clean faces are kept (identity + diversity
  survive Hungarian matching), grainy blends are pulled toward a smooth global mean
  face (de-grains *and* regresses a wrong identity to the SSIM-safe prior). Strictly
  more than a uniform mean-blend because it preserves the slots actually recovered.

---

## 5. The batch-128 wall (what does NOT work)

Confirmed by 4 independent lines that heavier gradient-inversion machinery does **not**
break batch-128 averaging here:

- SPEAR / CPA / ICA disaggregation — refuted (no rank knee at 128).
- Geiping cosine-matching (`geiping_attack.py`) — converges to mush past batch ~8–64.
- GradViT / GradInversion (Yin et al.) — designed for batch ≤ 8–64 with a BatchNorm
  prior we don't have; their **Eq.8 label rule** (`ŷ = argsort(min_m gradW_head[:,m])`)
  is wired into `recover_labels` but is soft at 128 imgs / 200 classes.
- EncGradInversion (batch 512) — needs last-layer label *disaggregation*, blocked by
  the same no-knee-at-128.

**Conclusion:** the analytic readout is the only clean signal; all remaining upside is
metric-level SSIM optimization on top of the best-possible base recon.

---

## 6. Reproduce / run

```bash
cd task3
source .venv/bin/activate                 # torch 2.12+cu130, skimage, scipy

# base recons per family -> recon/model{i}.pt
python mlp_attack.py                       # 1,4,5,8
python cnn_fullres_attack.py               # 10,12
python cnn_pooled_attack.py 2 3 7 --save   # 2,3,7  (improved inverter; writes recon/)
# ViT 9,11: cluster GPU job -> sbatch main.sh geiping_attack.py --model 9 --n 128 --iters 8000 --save

python postproc.py                         # recon/ -> recon_pp/model{i}.pt + viz/
python assemble.py                         # build+validate submission.pt (prefers recon_pp/)
python assemble.py --submit                # upload (reads ../.env API_KEY) — COORDINATE FIRST
```

Backups kept: `submission_958_backup.pt` (0.3035), `submission_pooled_only.pt`
(isolated pooled A/B), `recon/model{2,3,7}_prepool_backup.pt` (pre-fix base).

---

## 7. Discipline (hard-won)

- **Only the leaderboard is truth.** Local proxies (cosine, subspace residual,
  capture-%) do **not** track SSIM. Validate on the board.
- **Never bundle changes** — one variable per submission or attribution is impossible.
- **Grain is the SSIM killer; smooth beats faithful** (§3, §4).
