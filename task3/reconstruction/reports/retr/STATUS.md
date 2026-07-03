# Agent R — Retrieval & data provenance (Lever L1)

**Status:** NOT STARTED
**Owner code prefix:** `retr_*.py` · **Outputs:** `recon_work/outputs_retr/`, `recon_work/gallery/`

## The one question
Are the 1536 images verbatim images from a public face dataset (CelebA / CelebA-HQ / FFHQ)
at a fixed 64px crop? If YES → retrieve exact clean originals → SSIM ~1.0 per hit. Highest EV.

## Decisive test (do first, <1h)
- [ ] STEP 0 — provenance: inspect HF `SprintML/FL_Data_Reconstruction` + `SprintML/hackathon` for source dataset + preprocessing.
- [ ] STEP 1 — build 64px galleries under candidate crops (178→64, 148→64, 140→64, 128→64, native-64) for CelebA / CelebA-HQ / FFHQ.
- [ ] STEP 2 — membership test on best MLP recon (model1/4/5): (a) pixel/SSIM NN twin >0.95? (b) gradient-subspace residual ~0 (project candidate onto top-128 right-singular subspace of net.0 weight-grad; reuse `attack_mlp_svd_project.py`). YES/NO with evidence.
- [ ] STEP 3 (if YES) — retrieve all models via ArcFace/FaceNet cosine (FAISS), accept only when embedding + pixel + subspace-residual agree; write `retrieved_{model}.pt`.
- [ ] STEP 4 (if NO) — try FFHQ/LFW; if still no member passes, declare L1 dead, pivot to Agent G.

## Findings
_(fill in: YES/NO verdict, which dataset+crop, per-model #verbatim hits, mean residual)_

## Handoff to Agent B
_(which models have retrieved tensors ready; which slots need bank fill)_
