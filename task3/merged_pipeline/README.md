# Merged Pipeline — teammate's 0.3888 approach + our validated levers

A **self-contained, local** re-implementation of the teammate's best pipeline
(`test_pib2.pt` = 0.3888), with our three board-validated / principled improvements
integrated. Runs from `task3/gradients` + `task3/models`, no JURECA scratch deps.

```bash
cd task3/merged_pipeline
../.venv/bin/python build.py            # -> submission.pt + viz/*.png  (pools cached in work/)
../.venv/bin/python build.py --no-cache # force re-reconstruction
../.venv/bin/python build.py --submit   # build then upload (reads ../../.env)
```

## Stages (`build.py` orchestrates)

| Stage | File | What | Source |
|---|---|---|---|
| 1 | `reconstruct.py` | analytic ratio `gradW/gradB` per model; MLP→image, CNN→feature→conv-invert | teammate + **ours** |
| 2 | `reconstruct.py` | conv inversion (`invert_conv`, strided low-res latent) | teammate |
| 3 | `selection.py` | 128 diverse: noise-gate `is_real` → quality-rank → greedy dedup → farthest-point **face-bank** fill | teammate v3 |
| 4 | `postproc.py` | global Gaussian blur σ=1.0 + NLM h=0.05 (their biggest lever, +0.031) | teammate |
| 5 | `postproc.py` | weak/dead-model mean-face blend (ViT β0.5, m3 β0.45, m6 β0.40, global β0.20) | teammate |
| 6 | `postproc.py` | per-image adaptive blend `pib` toward blurred-self, γ=0.18 | teammate |

## Our three integrated levers (all in `config.py`)

1. **Aggressive low-rank MLP denoise** (`MLP_LOWRANK_K = {1:35,4:35,5:37,8:14}`).
   The teammate truncated the first-layer weight gradient at k=128 (kept ~90 noise
   directions → board tie). We project onto the **per-model 2×-noise-floor crossover**
   (~35, and 14 for the heavily-noised model8). Board-validated **+0.007** on our track.
   → `reconstruct.lowrank_denoise`.
2. **model2 tanh featureness fix**. model2's gradient is clean rank-128, but the
   teammate scored feature candidates with a relu-style penalty on negatives, which
   flags every legitimately-negative tanh feature as garbage → model2 treated as dead.
   We score against the tanh output range [-1,1]. Board-validated **+0.0062**.
   → `reconstruct.featureness`.
3. **CNN conv-nullspace chroma correction** — the teammate's *own* flagged "last
   untested recon lever". `model10` neon-chroma → desaturate to gray; `model12`
   sigmoid color-cast → per-channel mean neutralise. → `postproc.desaturate/colorcast_fix`.

## Files
- `common.py` — local paths, ported primitives (image_stats, TV, gaussian_blur, quality_score).
- `models.py` — `MLPNet` / `SimpleCNN` (ported; conv stride/pad inferred from feature_shape).
- `reconstruct.py` — stage 1-2 + our levers 1,2. Caches pools in `work/pools/`.
- `selection.py` — stage 3 (v3 diversity-first) + face bank from clean recoveries.
- `postproc.py` — stages 4-6 + our lever 3.
- `config.py` — every tunable knob (per-model betas, k, chroma, blur/NLM/pib params).
- `build.py` — orchestrator → `submission.pt`, validation, `viz/` grids.

## Leaderboard
First full build scored **0.3543** (submission 2247) — +0.038 over our prior 0.3166
track. The account's best is 0.3918; the gap is base-reconstruction quality, which
the cluster run (below) targets via higher-step, feature-mean-seeded conv-inversion.

## Reconstruction-quality levers (new, for closing the gap)
- **Feature-mean conv-init** (`config.UPSAMPLE_INIT`): seed the conv-inversion from
  the feature channel-mean (a low-freq face guess) instead of random noise. Round-trip
  validated +0.05..0.12 on the under-determined pooled models (2,3,7). `reconstruct.invert_conv`.
- **`config.CONV_STEPS`** (env `MP_CONV_STEPS`): 500 locally, **3000 on the cluster**
  for sharper full-res inversions. This is the heavy step GPU accelerates.

## Cluster run (the real test)
```bash
# on the cluster, from this folder (set MP_VENV to a CUDA torch env if not ../.venv):
sbatch job.sh                         # GPU, MP_CONV_STEPS=3000, --no-cache
# override data location / steps:
FL_DATA_ROOT=/path/to/FL_Data MP_CONV_STEPS=4000 sbatch job.sh
```
`job.sh` sets the SLURM headers (dc-gpu, cispahack reservation), points `FL_DATA_ROOT`
at the gradients/models dir, runs `build.py --no-cache` on GPU, and writes
`submission.pt`. Deps in `requirements.txt` (needs CUDA torch + scikit-image).

## Notes
- **Reconstruction (conv-invert) is the only heavy step.** It's cached after the first
  run; delete `work/pools/` or pass `--no-cache` to redo. If a GPU is available the
  conv-inversion auto-uses it (`torch.cuda.is_available()`), so heavy re-runs belong on
  the cluster.
- This is a faithful re-implementation of the documented algorithm, **not** a copy of
  the teammate's exact JURECA candidate banks — so the score may differ from 0.3888;
  the leaderboard is the arbiter. Our 3 levers are the intended lift over their base.
- Every knob is isolated in `config.py` for clean single-variable A/Bs (e.g. set
  `MLP_LOWRANK_K={}` to ablate lever 1, `DESATURATE={}`/`COLORCAST_FIX={}` for lever 3).
