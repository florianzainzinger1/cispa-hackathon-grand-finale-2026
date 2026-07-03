# Agent G — Generative prior (Lever L4)

**Status:** NOT STARTED
**Owner code prefix:** `gp_*.py` · **Outputs:** `recon_work/outputs_gp/`, `recon_work/runs/`

## The one question
Can a strong face prior rescue the hard models (ViT 9,11; strided CNN 2,3,7) beyond the diverse bank?
Reuse the r1/r2 no-plugin StyleGAN standup (`gan_prior.py`, `attack_stylegan.py`; force `impl='ref'`).
KNOWN: StyleGAN *latent-matching* = realistic but wrong-identity → LOSES on SSIM (dead end). Only
on-manifold *projection* of already-blurry recon helped (+0.0014). So the bar is high — measure before shipping.

## Tasks
- [ ] `gp_gifd.py` — GIFD-style feature-domain optimization on StyleGAN2-FFHQ-64: progressive latent→W+→intermediate-feature stages with a PER-STAGE ℓ1-ball projection (the fix for "high gradient-cosine but noise image") + TV + L2; infer labels; many restarts; keep best per-sample by gradient cosine. (Verify GIFD citation/benchmarks before trusting them.)
- [ ] `gp_dps.py` — DPS diffusion prior for ViT 9/11: plug-and-play with `ffhq_10m.pt`, forward operator = "compute this model's averaged gradient"; guidance step; multi-restart + align-and-fuse.
- [ ] ALWAYS ±3px align before scoring SSIM. Treat GFPGAN/CodeFormer as demo-only (can lower SSIM).

## Findings
_(per-model: prior output vs diverse bank on the proxy metric; keep or defer to bank)_

## Handoff to Agent B
_(which models have `gp_{model}.pt` that BEAT the bank; which defer)_
