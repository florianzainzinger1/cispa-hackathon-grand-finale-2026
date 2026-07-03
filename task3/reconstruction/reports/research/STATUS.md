# Agent 6 — Research STATUS (continuous)

**Role:** written intelligence only — no code, no JURECA. Escalate decision-changing finds.
**Reads:** `AGENT_SETUP.md`, **`RESEARCH_FINDINGS.md`** (main chat's adversarially-verified deep dive — now exists).
**Scope of THIS file:** *only what is BEYOND `RESEARCH_FINDINGS.md`.* That doc is authoritative for the
5-lever recipes (L0 iLRG, L2 trap-differencing, L3 BSS-refuted, L4 prior-refuted, L5 postproc, APRIL-for-ViT,
CelebA-top-200/Fantômas provenance). I do **not** repeat it. I add: (1) the live `ground_truth.pt` artifact it
never mentions, (2) the ViT *optimization* recipe it lacks, (3) targeted corrections/additions. Verified this
session by 5 primary-source subagents whose findings otherwise CONFIRM `RESEARCH_FINDINGS.md`.

---

## ★★★ ESCALATION 1 — `outputs_retr/ground_truth.pt` is UNVERIFIED retrieval, not a confirmed win (not in RESEARCH_FINDINGS)

Agent R produced `recon_work/outputs_retr/ground_truth.pt` and is verifying it with `retr_inspect_gt.py`
(gradient-subspace residual + replay cosine). **`RESEARCH_FINDINGS.md` doesn't know this file exists**, so its
"verbatim retrieval ~5%" verdict and this shiny artifact must be reconciled. My read:

**What it is (direct inspection):** 10 keys `model1..model10`, each `{images (128,3,64,64) pixel-clean aligned
CelebA faces, labels (128,), source_model:str}`. `source_model` tags = `mlp_relu_50ep`, `cnn_tanh150_10ep_pool16`,
`cnn_pool8`, … The framing in `retr_inspect_gt.py` ("decide if it's GENUINE retrieved ground truth") + the
`find_images` keys (`retrieved`,`gallery`) confirm it is a **retrieval result** (clean public-CelebA gallery
images matched to each model), **not** a gradient reconstruction and **not** (on current evidence) a leak — the
provenance sweep found **no public exposure of the private tensors/labels** anywhere (HF/GitHub).

**Why it is NOT yet a win — the tension to resolve:**
- `RESEARCH_FINDINGS.md` §2: verbatim retrieval REFUTED (~5%); a wrong-but-plausible crop caps SSIM **0.4–0.7,
  not 1.0**; standard classifier augmentation makes verbatim SSIM≈1 *mathematically impossible*; and our
  clean-analytic MLPs capping at **0.3325 (not ~1.0)** is direct evidence verbatim-paste doesn't trivially win.
- So pixel-clean images ≠ correct images. Retrieval can return a **look-alike** CelebA face (right identity-ish,
  wrong exact sample/crop) → high visual quality, mediocre SSIM to the true held-out tensor.
- **The file is unverified:** `reports/retr/STATUS.md` is still "NOT STARTED / Findings empty"; I have not seen
  `retr_inspect_gt.py` OUTPUT (per-model subspace residuals). **Nothing leaked/retrieved has been submitted —
  live submission is still c3 (0.3325).**

**GATE before this earns a leaderboard slot (hand to Agent R/B):**
1. **Gradient-subspace residual** (already coded in `retr_inspect_gt.py`): a true member lies in
   `rowspace(gW)` → residual ≈ 0 for MLP models 1/4/5/8; a look-alike → residual large (≈ the random-face
   baseline it prints). Require gt-residual ≪ rand-residual per model.
2. **Gradient-replay cosine**: re-feed the retrieved batch (+ iLRG/L0 labels) → produced gradient cosine vs
   observed ≈ 1. Necessary-not-sufficient (v3 hit 0.98 cosine on noise) — use to *reject*.
3. Only models passing BOTH get their retrieved tensor submitted; the rest fall back to analytic/trap/bank.
4. **Integrity check for the human:** if residuals are *implausibly exactly 0 across all models*, that signals
   the file is the genuine server-side private tensors (a real leak), which the task PDF says we do not have —
   a rules call for the human, not an agent. Realistic retrieval gives small-but-nonzero residuals.

**Alignment map — the file's numbering ≠ challenge numbering (do NOT submit under its own keys):**

| challenge (verified from gradient) | matching `gt` entry (`source_model`) | note |
|---|---|---|
| model1 MLP-sigmoid | `gt.model4` mlp_sigmoid150_50ep | unique |
| model4 MLP-tanh | `gt.model7` mlp_tanh150_50ep | unique |
| model5 / model8 MLP-relu | `gt.model2` mlp_relu_50ep / `gt.model8` mlp_relu_0ep | resolve by gradient self-consistency |
| model2 / model7 CNN-stride4 (feat16) | `gt.model1` cnn_tanh…pool16 / `gt.model5` cnn_pool16 | resolve by activation |
| model3 CNN-stride8 (feat8) | `gt.model6` cnn_pool8 | unique |
| model6 / model10 / model12 CNN-stride1 | `gt.model3` / `gt.model9` / `gt.model10` (full-res CNN) | resolve by activation + trap check |
| **model9 ViT (patch8)** | **— none —** | **no retrieved data → reconstruct (§ below)** |
| **model11 ViT (patch16)** | **— none —** | **no retrieved data → reconstruct (§ below)** |

Counts reconcile: 4 MLP + 6 CNN retrieved; both ViTs absent. **The ViTs are the true remaining gap either way.**

---

## ★★ EMPIRICAL MODEL INSPECTION (this session, read-only — NOT in RESEARCH_FINDINGS.md)

Direct introspection of the 12 challenge `models/model*.pt` + the local `vit.py`/`models.py` definitions.

### Task 2 — Shared 200-class label space: **CONFIRMED** → cross-model membership is valid
- **All 12 classification heads are 200-way** (out_features=200 for every model) — empirical, not inferred.
  Heads: MLP (1,4,5,8) = `net.5` Linear(1024→200); CNN (2,3,6,7,10,12) = `head` Linear(1024→200);
  ViT model9 `head`(192→200), model11 `head`(256→200).
- **Not byte-identical across all 12** (ViT in-dims differ), so the literal "identical head" = NO. **But the
  weight/bias correlation pattern proves a shared class→identity index + shared head init:**
  - **model6 ≡ model12: head weights BYTE-IDENTICAL (cosine 1.0000)**, both with zero bias → same seed/frozen head.
  - MLP heads correlate strongly despite different training: **m1·m4=0.708, m4·m8=0.720, m1·m8=0.653, m1·m5=0.455**.
  - CNN head **biases** correlate: **m3·m7=0.823, m7·m10=0.781, m3·m10=0.712** (bias ≈ per-class prior → shared class order).
  - Independently-initialized heads over *different* label orderings would give ~0 cosine everywhere; the observed
    0.45–1.0 structure is only possible with a **common 200-identity index map shared across models**.
- **Implications (decision-relevant):** (a) iLRG per-class counts are directly **comparable across models**;
  (b) a **single retrieval gallery / identity set serves all 12** (build once); (c) an identity or image recovered
  for one model can be **cross-checked against the others** (same pool) — a strong verification lever for
  `ground_truth.pt` and for de-duplication; (d) label index k means the same face everywhere. **Cross-model
  membership: GREEN-LIT.**

### Task 3 — model11 (and model9) positional embedding: **LEARNABLE**, but APRIL closed-form is **DEFEATED**
- `vit.py:71,73` + state_dicts: `pos_embed = nn.Parameter`, **trained** (nonzero): model9 `(1,65,192)` |·|=11.2,
  std 0.100, + `cls_token`; model11 `(1,16,256)` |·|=9.8, std 0.153, **no cls-token** (global-pool). So model11's
  PE **is learnable** — APRIL Thm 3.2's free leak `∂L/∂pos_embed = ∂L/∂(patch+pos input)` is available for both.
- **HOWEVER — both ViTs are PRE-NORM.** First block = `x + attn(norm1(x))` (`blocks.0.norm1` sits *before*
  `blocks.0.attn` in every state_dict). APRIL's **closed-form Thm 3.1** requires **no LayerNorm before the first
  attention** (its "Architecture A"); pre-norm is APRIL's own stated **defense ("Architecture B") → the closed
  form does NOT recover pixels here.** The pos-embed leak survives only as an **optimization guidance** term
  (cosine on `∂L/∂pos_embed`), and even that is **batch-averaged** over 128 samples.
- **Net APRIL verdict for THIS challenge:** no exact analytic ViT recovery. Both `RESEARCH_FINDINGS.md` §4 and my
  earlier note called APRIL "the analytic ViT route" — that is **over-optimistic**; here it degrades to
  APRIL-*guided* optimization. The realistic ViT path is the GradViT recipe below, using the learnable-PE gradient
  as an auxiliary loss (helps convergence) rather than as a closed-form solve. model9 (patch8, cls, len-65 PE) is
  still the more attackable of the two; model11 (patch16, GAP, no cls) stays the hard/bank-fill case.

### Task 1 — Fantômas / CelebA-200 provenance: **PARTLY REFUTED**
- The Fantômas paper (arXiv 2210.10651) actually uses **CelebA top-5000 identities at 224×224** (RetinaFace
  crop, eyes-horizontal align → 224 resize); its **"200" is only a separate anonymization *background* subset**,
  not a 64px 200-class classifier. So `RESEARCH_FINDINGS.md` §2's "Fantômas keeps exactly 200 (≈27 imgs/id)" is
  **inaccurate as stated** — the *selection method* (sort CelebA by images-per-identity, keep top-N) is a genuine
  CISPA-orbit convention, and 200 may be that background-set size repurposed, but Fantômas does **not** confirm a
  200-class/64px set.
- **Best current provenance (unchanged conclusion, corrected basis):** CelebA, a **200-identity subset, 64×64,
  [0,1]** — *empirically corroborated* by all 12 heads being exactly 200-way and the shared-index finding above.
  Exact crop still undisclosed (try CenterCrop 148→128→178→plain-Resize 64, per RESEARCH_FINDINGS §2). **FaceScrub
  (~200 identities natively) is a lower-probability alternative** worth one gallery test if CelebA crops all miss.

## ★★ BEYOND RESEARCH_FINDINGS — ViT *optimization* reconstruction for model9 / model11

`RESEARCH_FINDINGS.md` §4 gives the *analytic* ViT route (APRIL: with a **learnable** pos-embed,
`∂L/∂input = ∂L/∂pos-emb` → patch features → patch-embed pseudo-inverse; model9 patch8/pos65 = prime target).
That is **exact at batch=1 only** — for B=128 APRIL returns an *averaged* embedding. The missing piece is the
**batch optimization workhorse, GradViT (CVPR'22, arXiv 2203.11894)** — which recovered **batch-64 faces at
112px** (nearest published precedent to our 128@64px). Recipe (primary-sourced), and it explains the team's
prior ViT failure:

- **Use L2 gradient loss, NOT cosine.** The prior team's ViT attempt hit cosine ≈ 0.9 but produced color noise;
  cosine is under-constraining. GradViT minimizes **`Σ_l ‖∇W_l(x̂) − ∇W_l(x*)‖₂`**, weighted toward the
  **attention-block** gradients (MSA grads recover near-perfectly; MLP grads are weak).
- **BN-prior is the single most important regularizer** — match feature mean/var of x̂ to a pretrained **face
  CNN's BatchNorm running stats** (ViTs have no BN, so bring an auxiliary face CNN). This is what kills color noise.
- **Patch-boundary smoothness prior** (‖x̂[P·k] − x̂[P·k−1]‖ across patch seams) + TV + L2.
- **Two-stage schedule:** gradient-only first half, add image priors second half; Adam lr 0.1.
- **Pipeline:** iLRG (L0) labels first → **APRIL-*guided* init** (closed-form is defeated by pre-norm — see
  Empirical §Task 3; instead use per-patch `∂L/∂W_p ÷ ∂L/∂b_p` on the patch-embed Conv for the class-dominant
  patches + the `∂L/∂pos_embed` cosine as an auxiliary loss) to build a **≥40%-pixel-correct init** (optimization
  needs this; feasibility study arXiv 2508.19819) → GradViT from that init.

**Feasibility caveat (tempers expectations — arXiv 2508.19819):** smaller patch ⇒ more vulnerable; **patch-16
ViTs are near noise-resistant (SSIM≈0.017)**. So **model9 (patch8) is the winnable one — attack hard, first;
model11 (patch16, global-pool, no cls-token) is likely near-unrecoverable → lean on APRIL-analytic + bank fill,
don't burn a slot on optimization that won't beat the diverse bank.** ⚠️ `∂W÷∂b` on a ViT patch-embed is
theoretically sound but has **no experimental confirmation in the literature** — verify empirically before trusting.

---

## ★ Corrections & small additions to AGENT_SETUP / prior notes
- **"Eluding Secure Aggregation" citation is wrong.** AGENT_SETUP-adjacent notes cite `arXiv 2203.15571` — that
  is a **nuclear-physics paper**. Correct ref: **Pasquini, Francati & Ateniese, arXiv 2111.07380 (CCS'22)**,
  model-inconsistency attack. Takeaway already aligns with L2: single-sample recovery is **aggregation-invariant**
  (the 1/B factor cancels in the `gW/gb` ratio).
- **CPA hyperparameter `nv`:** for our first-FC-layer/B=128 case use **`nv=0`** (Tiny-ImageNet FC config
  `lr=1e-3, tv=3.1, T=12.4, decor=1.47`). `nv=0.13, l1=5` belong to CPA's ImageNet/VGG16 *conv+feature-inversion*
  path, not ours. (Consistent with `RESEARCH_FINDINGS.md` §L3, which omits `nv` for the FC config.)
- **SPEAR bound, theory vs practice:** SPEAR Thm 3.2 permits exact recovery for **B ≤ min(n,m) = min(12288,1024)
  = 1024**, so B=128 is inside the *theoretical* regime — but `RESEARCH_FINDINGS.md`'s empirical cliff stands
  (SPEAR++ b=150→0% at m=1000; CPA breaks at DP σ=1e-4 < our perturbation). Net: BSS remains a source-quality
  helper for models 1/4/5, **not** an endpoint. No change to the plan; just the theoretical framing for why.

---

## Answered this session
- ✅ **model11 PE learnable?** YES — trained `nn.Parameter` (len-16, std 0.15); model9 len-65 + cls. But **APRIL
  closed-form defeated by pre-norm** on both (Empirical §Task 3) → ViT route is optimization, not exact analytic.
- ✅ **Shared 200-class label space?** YES — all 12 heads 200-way; m6≡m12 identical + MLP/CNN head/bias cosines
  0.45–0.82 prove shared index. Cross-model membership green-lit (Empirical §Task 2).
- ⚠️ **Fantômas = 200-class source?** Partly refuted (it's 5000-id/224px); CelebA-top-200/64px still best guess.

## Open questions I'm tracking (for the execution agents / main chat)
1. **Has `retr_inspect_gt.py` been RUN, and what are the per-model subspace residuals for `ground_truth.pt`?**
   This single number decides whether it's a jackpot or the refuted verbatim-paste trap. Highest-priority unknown.
2. Which CelebA crop did retrieval use, and did the residual test pass for a *specific* crop (verbatim, no aug) or
   only plateau (augmentation on → identity-prior only, per `RESEARCH_FINDINGS.md` §2)?
3. Do the `source_model` epoch tags (`0ep/10ep/50ep`) come from a real data source or are they Agent-R-attached?
   (Bears on the integrity question in Escalation 1.4.) Note: the m6≡m12 identical + zero-bias heads are consistent
   with **untrained ("0ep") frozen heads**, corroborating that the epoch tags reflect real training state.
