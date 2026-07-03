# r1 / r2 — StyleGAN Generative-Prior Gradient Inversion (risk branch)

High-risk / high-reward branch: use a pretrained face generator as an image prior
so the gradient-inversion search happens on the face manifold instead of raw
pixels. Built and run as a new **additional candidate source**; the existing
analytic/v1/v2/v3 pipeline is untouched.

## 1. What `risk.pdf` said (used as the technical basis)
- Generator priors help precisely where DLG / Inverting-Gradients / breaching give
  noisy/blurry/half-right images: search in **StyleGAN W/W+ latent space** (or a
  diffusion trajectory) rather than pixels → more stable, more natural images.
- Recommended staging: **cheap analytic checks first**, then **StyleGAN latent
  search**, then **latent+pixel hybrid**, then diffusion (only hardest models),
  then a learned inverter (biggest gamble). For a single creative strategy the PDF
  picks **StyleGAN latent (+ optional pixel residual)**.
- Loss: **cosine gradient matching (primary, scale-invariant) + small L2 + TV +
  latent prior ‖w−w_avg‖²**; BN-stat term only if the victim has BatchNorm (ours
  do **not**). Init at `w_avg` with multiple restarts.
- **Batch handling for 128**: don't jointly optimise 128 latents (collapses);
  iterative small subsets (k=4..16) + diversity + freeze-best. Labels are often
  wrong → use soft/alternating labels, not hard ones.
- Failure modes it explicitly warns about — **domain mismatch, realism-over-
  fidelity, batch collapse, wrong labels, high-cosine-but-wrong-image** — all of
  which we hit and had to engineer around (see §9–11).
- Concrete resources: NVlabs `stylegan2-ada-pytorch`, weights `ffhq.pkl` /
  `stylegan2-celebahq-256`, cosine+TV objective, pseudocode for
  `reconstruct_with_stylegan`. We followed this closely.

## 2. Extra research
- Subagent A mapped the existing pipeline: candidate schema, selection, labels,
  per-model status, and the pivotal fact that **v3 already tried pixel-space
  cosine matching and it produced pure noise** (high cosine ≠ good image at
  batch-128 with defended gradients — only 47–78 % of gradient energy is in the
  top-128 subspace). That negative result is exactly the gap a manifold prior
  fills.
- Subagent B nailed the generator-standup recipe for the JURECA stack (torch
  2.12+cu130, **no ninja / nvcc / torchvision**): NVlabs runs in **no-plugin
  pure-PyTorch mode** because `bias_act`/`upfirdn2d` fall back to `_ref` and
  `conv2d_gradfix` is version-gated to native `F.conv2d` (which supports the
  create_graph double-backward the attack needs).

## 3. Logic designed
Victims all take `(B,3,64,64)` in `[0,1]` → `(B,200)` logits (CE loss). Two uses
of a **CelebA-HQ-256 StyleGAN2** prior:
- **(A) latent gradient-matching** — optimise latents `w` so `victim` gradient of
  `G(w)` matches the observed gradient (cosine + L2 + TV + latent-prior +
  diversity), soft learnable labels, many restarts; select 128 diverse by greedy
  max-min. **The manifold keeps outputs as faces, not noise.**
- **(B) manifold projection** — image-space GAN-invert existing blurry
  reconstructions onto the StyleGAN manifold (label-free, per-image, low-risk).
- **r2 fix**: r1's W+ latent search *gamed the cosine with degenerate saturated
  images*. r2 optimises in the well-conditioned **W space** with a **truncation
  clamp** (L2 ball around `w_avg`, radius auto-calibrated from sampled-W stats),
  strong latent prior, no latent-space diversity term → clean faces at the same
  replay cosine.

## 4. Files / folders created
Code (in `task3/reconstruction/`, synced to JURECA `recon_work/code/`):
- `gan_prior.py` — load NVlabs G in no-plugin mode; differentiable `[0,1]` 64×64
  synth wrapper; W/W+ init; `w_radius` truncation calibration.
- `attack_stylegan.py` — losses, latent gradient-matching, manifold projection,
  gradient-replay proxy, diverse selection, truncation clamp, CLI (smoke/latent/
  project).
- `job_r1_stylegan.sh` — sbatch job (A100, no-plugin PYTHONPATH, env).
- `r1_build_submission.py` — assemble a submission preferring r2→r1→base per model.

Outputs (fresh trees, nothing existing touched):
- `recon_work/runs/r1/<model>/` — latent (model2/9/11) + project (model2/3/7).
- `recon_work/runs/r2/<model>/` — clean W-space+truncation latent (model2/9/11).
- `recon_work/runs/submissions/submission_r1r2.pt` (+ montage, provenance).
- `recon_work/runs/{repos/stylegan2-ada-pytorch, weights/celebahq256.pkl, pylibs}`.

## 5. Code changed
Only **new** files added (list above). No existing pipeline file was modified. The
challenge `.venv` was not modified (deps installed to an isolated `runs/pylibs`
via `uv pip install --target`).

## 6. Commands run (representative)
- Feasibility probe (internet/toolchain), `git clone` NVlabs, download+verify
  `celebahq256.pkl`.
- Import/construct test on torch 2.12 → PASS.
- `sbatch job_r1_stylegan.sh --mode smoke --models model2` → double-backward OK.
- `--mode latent` (r1: W+, l_w=1e-3) for model9/11/2; `--mode project` for
  model2/3/7.
- r2: `--mode latent --no-w-plus --trunc=-1 --l-w 5e-3 --l-div 0` for model2/9/11.
- `python r1_build_submission.py` + `validate_submission.py` → VALID.

## 7. Where r1/r2 outputs are saved
`recon_work/runs/r1/`, `recon_work/runs/r2/`, `recon_work/runs/submissions/`.
Each `<model>/selected_128.pt` = `{"images": (128,3,64,64) float32 [0,1], ...}`,
matching the pipeline schema; grids + metrics.json alongside.

## 8. v1/v2/v3 unchanged
Yes. Nothing was written to `candidates/`, `candidates_v2/`, `candidates_v3/`,
`outputs/`, `outputs_v2/`, `outputs_v3/`, `submission*.pt`, or the challenge data.
All r-branch outputs live under `recon_work/runs/`. `candidates_v3/` was only
**read** (as projection targets).

## 9. Promising models
Metric = replay cosine of the selected-128 aggregate gradient vs observed
(higher = better; random-face baseline in parens):
- **model2 (CNN, tanh, 0 analytic rows): r2 replay 0.666 (random −0.339)** —
  clean, diverse, realistic faces AND strong gradient explanation. Best result.
- **model9 (ViT, 0 rows): r2 replay 0.358 (random −0.249)** — clean diverse faces
  replacing v3's duplicated shared bank.
- **model3, model7 (strided CNN, blurry): project** — clean on-manifold faces
  (recon MSE 7e-4 / 1.2e-3); modest but reliable sharpening.

## 10. Hard models
- **model11 (ViT, patch16/global-pool): r2 replay 0.054 < random 0.147** — the
  gradient match fails and faces stay partly colour-tinted; the prior does **not**
  beat random/bank here. Genuine negative.
- General ceiling: gradients are **defended/noised**, so even good per-subset
  cosine (0.6–0.87) only partially transfers to the full-128 aggregate. High
  cosine can be gamed by degenerate latents (r1 W+ for model2/11) — the truncation
  clamp is what turns that into real faces.

## 11. r2 done; r3 directions
r2 (W-space + truncation) already fixed the r1 degeneration. Next:
- **Latent+pixel hybrid** (PDF Phase 2): freeze `w`, add a bounded pixel residual
  δ with TV/LPIPS-to-G(w) → sharper local detail on model2/9.
- **Looser projection** for model3/7: match only low-freq structure so the prior
  fills texture instead of copying the blur.
- **model11**: try FFHQ-1024 prior, per-layer W+ within a tighter truncation, or a
  learned inverter; if still failing, keep diverse random prior faces (cleaner
  than the shared bank).
- Consider a proper **iterative-subset freeze scheme** and **BN-free label
  alternation** to lift the full-128 replay for the ViTs.
