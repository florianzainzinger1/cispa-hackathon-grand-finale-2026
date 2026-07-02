# Task 3 — FL Data Reconstruction from Gradients

Reconstruct 128 private 64×64 face images for each of 12 models (1536 total) from
one client gradient per model. Scored by per-model Hungarian one-to-one SSIM, so
**both quality and diversity matter**.

## Data (on JURECA, not in this repo)

`/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/{models,gradients}/model{i}.pt`
- `gradients/model{i}.pt` → `{gradients: {param→grad}, family, activation, feature_shape, batch_size}`
- `models/model{i}.pt` → plain `state_dict`

Set `FL_DATA_ROOT` / `FL_WORK_ROOT` to override paths. Workspace outputs go to
`recon_work/` (candidates, reports, outputs) — kept out of git.

## Model triage (the whole game)

| models | family | leak path | quality |
|---|---|---|---|
| 1,4,5,8 | MLP | `net.0` is `Linear(12288→1024)`; `gradW[r]/gradB[r]` reshapes straight to (3,64,64) | high |
| 6,10,12 | CNN (stride-1 conv) | `fc1` analytic → 8×64×64 feature → invert conv → RGB | high |
| 2,7 | CNN (stride-4) | feature 8×16×16 → smooth low-res inversion | low (blurry) |
| 3 | CNN (stride-8) | feature 8×8×8 | low (blurry) |
| 9,11 | ViT | no analytic leak; gradient matching gives incoherent patches → **face-bank fallback** | weak |

Analytic ratio = weighted average of a linear layer's inputs; ReLU gating isolates
single samples, so many of the 1024 neurons each reconstruct one private image.
Clustering the neuron pool to 128 (best/robust-mean per cluster) recovers a
diverse, de-noised set — essential for the one-to-one SSIM matching.

## Pipeline

```
inspect_models.py     # architecture + attack-path triage → reports/
attack_analytic.py    # gradW/gradB candidates (MLP images, CNN features)
attack_conv_invert.py # invert first conv: CNN feature → RGB image
infer_labels.py       # iDLG-style label multiset from head-bias gradient
attack_opt.py         # gradient matching (cosine+TV) for ViT [GPU: job_recon.sh]
postprocess.py        # cluster → 128 diverse, robust-mean denoise
face_bank.py          # diverse real-face bank (ViT fallback)
build_submission.py   # assemble + validate outputs/submission.pt
validate_submission.py# standalone re-check of a saved .pt
run_pipeline.sh       # runs the CPU stages end-to-end
```

CPU stages (inspect/analytic/conv-invert/postprocess/build) run on the login
node in minutes. ViT gradient matching runs on a GPU via `sbatch job_recon.sh
model9 model11` (env: `FL_ITERS`, `FL_BASE_RES`, `FL_INIT=bank`, `FL_TV`).

## Submission format

`.pt` dict, keys `model1..model12`, each `Tensor(128,3,64,64)` float32 in `[0,1]`,
finite, no exact duplicates. Endpoint: `POST http://<SUBMIT_SERVER>/submit/21-fl-audit`
with header `X-API-Key` (key NOT stored in this repo).
