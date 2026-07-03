# v3 — Coverage-First Analytic Reconstruction with a Diverse Face Manifold

v3 is a **different method** from v1 (analytic ratio → KMeans → quality-weighted
robust-MEAN → noise-pad) and v2 (analytic ratio → confident single-sample
`tau`-clustering → blend with v1). It was designed after reading the task's three
referenced papers (Boenisch *trap weights*, Zhu *DLG*, Geiping *Inverting
Gradients*) plus the two strategy dossiers in `info/papers/`.

## What v3 does differently

1. **Explored the optimization family (DLG / Inverting-Gradients) and rejected it.**
   `v3_gradmatch.py` implements the full Geiping recipe: match the observed
   client gradient with a **cosine** (magnitude-invariant) loss + TV, **signed
   Adam** with LR-decay, box-constrained to `[0,1]`, **learnable soft labels**
   (DLG-style, so no per-image label↔image pairing is needed), warm-started from
   the analytic candidates / face bank.
   **Result (verified on GPU):** the gradient cosine climbs to **0.76 (MLP), 0.81
   (CNN), 0.98 (ViT)** — yet every refined image is **pure RGB/colour noise**.
   At batch-128 the averaged-gradient inverse problem is under-determined and the
   optimiser finds a high-cosine noise solution. **High gradient cosine ≠ good
   image.** So gradient matching is *not* used as a reconstruction source; the
   module remains as the documented negative result that motivates the design.

2. **Sharp, diversity-first selection instead of blurring robust-mean.**
   `v3_select.py` pools each model's analytic outputs — v1's curated
   `selected_128`, the full CNN `conv_invert` pool, and (MLP only) the analytic
   single-neuron faces using `images_norm` (correct per-image min-max, which the
   raw clamp broke for the tanh/sigmoid MLPs) — then:
   * drops **noise** slots (finite, mostly in-range, and total-variation below a
     face/noise cut — RGB noise has TV≈0.33, faces ≈0.05–0.15);
   * **quality-ranks + greedy-dedups** to keep up to 128 *sharp, distinct* faces
     (no averaging → no blur), which is what the Hungarian one-to-one SSIM metric
     rewards;
   * applies a **TV-gated denoise** that smooths only the noisy reconstruction
     slots (clean faces stay sharp).

3. **Coverage-complete — never noise.** Any shortfall is filled with the most
   **diverse** faces (farthest-point sampling) from a rebuilt clean face bank
   (`v3_bank.py`: the sharpest, noise-filtered, quality-ranked reconstructions
   across the six strong models, deduped to 128 diverse faces). v1 padded
   shortfalls with RGB **noise** (≈0 SSIM); v3 never does — this is the single
   biggest fix (e.g. model6 went from ~50 % noise to 46 real + 82 diverse faces).

## Per-model outcome (`candidates_v3/_select_summary.json`)

| models | family | v3 source | notes |
|---|---|---|---|
| 1,4,5,8 | MLP | 128 sharp analytic faces | `analytic_norm` fixes tanh/sigmoid scaling |
| 10,12 | CNN (stride-1) | 128 conv-invert faces | ~365–383 clean available |
| 6 | CNN (stride-1) | 46 conv-invert + 82 bank | only 48 clean → bank fills, **no noise** |
| 2,7 | CNN (stride-4) | 128 low-res faces | blurry but real |
| 3 | CNN (stride-8) | 128 low-res faces | weakest (8×8 feature) |
| 9,11 | ViT | 128 diverse bank faces | analytic mixed + gradient-matching = noise |

## Files & run order

```
v3_gradmatch.py   # optimization attempt (diagnose|refine) — GPU; negative result
v3_bank.py        # clean diverse 128-face manifold from strong models
v3_select.py      # sharp quality-dedup selection + never-noise diverse fill
v3_build.py       # assemble + validate outputs_v3/submission_v3.pt
job_v3.sh         # sbatch wrapper for the (rejected) GPU gradient-matching
```

CPU stages (`v3_bank → v3_select → v3_build`) run on the login node in seconds.
Output: `recon_work/outputs_v3/submission_v3.pt` (valid, 12/12, no noise).
Never overwrites v1's `outputs/submission.pt` or `submission_v1_bank.pt`.
