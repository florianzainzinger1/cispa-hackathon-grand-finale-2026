# Agent S — Batch source separation (Lever L3)

**Status:** NOT STARTED
**Owner code prefix:** `bss_*.py` · **Outputs:** `recon_work/outputs_bss/`

## The one question
Can we separate each honest model's averaged first-layer gradient into the 128 INDIVIDUAL faces?
(Our ratio+cluster pipeline only recovers single-sample-dominant neurons and blends the rest.)
Note: naive FastICA(128) UNDER-converges — the full recipe is required.

## Tasks
- [ ] `bss_cpa.py` — Cocktail-Party Attack on MLP net.0 (models 1,4,5,8): whiten gW to top-128 subspace (PCA), learn 128×128 orthogonal unmixing by Adam maximizing negentropy(log-cosh) + TV + decorrelation + sign(nv) + L1; fix scale with bias gradient; fix sign by in-[0,1]/low-TV orientation; seed from analytic-ratio rows. (Verify hyperparams against the paper before trusting specific numbers.)
- [ ] `bss_spear.py` — SPEAR/SPEAR++ style: SVD gW → L,R; argmin ‖L q‖₀ on unit sphere via ℓ1/LogCosh + Riemannian-Adam, many inits, filter by sparsity-match; scale via bias gradient.
- [ ] CNN (6,10,12 and strided 2,3,7) — run BSS on fc1 gradient → 128 conv-feature maps → invert conv (`attack_conv_invert.py`) → RGB.

## Findings
_(which method wins per model; #distinct faces recovered; grad self-consistency + subspace residual)_

## Handoff to Agent B
_(which models have `bss_{model}.pt` and whether they beat c3's source on the proxy)_
