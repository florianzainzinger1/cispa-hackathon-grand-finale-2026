# Task 3 — Final Push Coordination Hub (target: beat 0.48)

**This file is the shared brain for a multi-chat agent push.** Main agent owns/edits it.
Worker agents READ Parts I–VI for strategy + their brief, and READ Part VII for ops.
Status: our best **0.3325** (c3). Leader **0.48**. Final rank = hidden **70%**
split → every method here must *generalize*, not fit the public 30%.

---

## PART 0 — LIVE SETUP STATE (main agent maintains — read this first)

**Setup completed 2026-07-02 (this session). Environment is GO.**
- JURECA reachable via WSL ControlMaster socket (verified, jrlogin10). Challenge data present
  (12 gradients + 12 models), venv intact (`torch 2.12.1+cu130`), Slurm reservation `cispahack` live.
- **Namespace scaffolding CREATED on JURECA:** `recon_work/outputs_{retr,trap,bss,gp,final}/`,
  `recon_work/gallery/`, `recon_work/reports/{retr,trap,bss,gp,build}/`, `recon_work/runs/pylibs/`.
- **Canonical baseline seeded:** `recon_work/outputs_final/submission_current_best.pt` = a COPY of
  c3 (`outputs_v3/submission_v3.pt`, scored **0.3325**). Agent B owns/advances this file; it is
  always the valid fallback. Originals `outputs_v3`, `outputs`, `_c4/_c5` are frozen — never overwrite.
- **Code synced:** all local `task3/reconstruction/*.py|*.sh` → `recon_work/code/` (CRLF-stripped).
- **Sync discipline (IMPORTANT, prevents collisions):** each agent scps ONLY its own prefix files
  (`retr_*.py`, `trap_*.py`, `bss_*.py`, `gp_*.py`, `merge_*.py`/`proxy_*.py`) into `code/` — NEVER
  `scp *.py` (that clobbers other agents). Build new prefixed files; don't edit shared modules
  (`utils.py`, `models.py`, `vit.py`, `attack_*.py`) — copy-and-rename if you need to change them.

**Launch decision (evidence-weighted, single-A100 aware):**
| agent | lever | status | why |
|---|---|---|---|
| R (retrieval) | L1 verbatim public-data | **LAUNCHED** | highest EV; decisive YES/NO test; could explain 0.48. Login-node/IO. |
| T (trap/exact) | L2 Robbing-the-Fed differencing | **LAUNCHED** | high EV, verbatim on leaky models; light analytic, independent of R. |
| S (BSS) | L3 SPEAR++/Cocktail-Party ICA | **LAUNCHED** | best pure-recon lever; GPU work self-serializes on Slurm. |
| B (build/proxy) | L5 + integration | **LAUNCHED** | foundational proxy harness; keeps `submission_current_best.pt` valid. |
| G (gen prior) | L4 GIFD/DPS | **LAUNCHED** (human-directed) | GIFD per-stage ℓ1-ball projection + DPS are a genuinely NEW angle vs the failed StyleGAN-latent; briefed to measure honestly on the proxy and DEFER to bank unless it clearly wins. Shares the single A100 with S. |

Prior confirmed DEAD-ENDS (do not re-run): batch-128 pixel gradient-matching → pure noise (all
families); StyleGAN **latent-matching** → realistic-but-wrong-identity, −0.0093 on leaderboard;
"recover more noisy faces" (c4/c5) → net-zero/negative (SSIM punishes grain > wrong-identity bank).
The ONLY positive generative lever found was on-manifold **projection** of the blurry model3/7 recon
(+0.0014). See Part III.

---

## PART 0.5 — STRATEGY UPDATE (deep-research results — read after PART 0, overrides lever ranks)

The main chat's adversarially-verified research workflow completed. Full doc:
**`task3/reconstruction/RESEARCH_FINDINGS.md` — every agent reads it once.** Three plan-changing results:

1. **NEW lever L0 — iLRG label-MULTISET recovery** (ICLR'23, code `BUAA-CST/iLRG` **master** branch).
   From the 200-way head weight+bias gradients, a `(C+1)×C` pseudoinverse recovers the exact per-class
   COUNTS `k_j` at B=128. Free, universal, zero-risk. Gives: per-label Hungarian blocks, the number of
   distinct identities (→ average duplicates for √k denoise), and class priors. **Run FIRST on all 12.**
2. **L2 (trap-differencing) is now the PRIMARY reconstruction lever** (after L0). Detection checklist
   (`dup_frac`,`w_rank1`,`bias_monotone`,`frac_neg≈0.5`,`dead_bias`) routes each model to Fowl-imprint
   *consecutive-bin* differencing vs Boenisch-trap *ratio*; **intra-cluster differencing**
   `(gW_i−gW_j)/(gb_i−gb_j)` cancels shared noise (attacks grain directly). Expect 60–100/128 verbatim
   on flagged models.
3. **L3 (BSS) and L4 (prior) are REFUTED as standalone reconstructors at B=128 — DEMOTED to support.**
   CPA breaks below our noise level; SPEAR++ has a 0%-accuracy cliff at exactly m=1024/b=128 and dies on
   GELU model8; GIFD's 0.577 is a B≤4 number and our StyleGAN-latent already scored −0.0093. **Neither
   earns a submission slot as an endpoint.**

**Revised lever stack:** `L0 iLRG (new) → L2 trap-differencing (primary) → L1 retrieval (test first,
don't assume, ~15–20% it helps some slots) → L5 postproc (align+blur+dup-avg, reliable +0.05–0.15) →
L3 BSS (source-only) → L4 prior (projection-only)`.

**ONE-OWNER-PER-MODEL STRUCTURE (this OVERRIDES the Part V chat briefs for agent3 & agent5).**
The overlap problem: agent2 and agent3 were both attacking the MLP first-layer gradient. Fixed by
giving each model exactly ONE reconstruction owner. agent1 is a global *override* (not a competitor);
agent4 is the sole integrator. **Main goal = ViT (9,11) → two agents now dedicated to it.**

| models | family | OWNER | approach | others must NOT touch |
|---|---|---|---|---|
| 1,4,5,8 | MLP | **agent2** | trap-detect → Fowl/Boenisch differencing → intra-cluster denoise | agent3/agent5 |
| 6,10,12 | CNN s1 | **agent2** | fc1 imprint check → analytic feature → conv-invert | agent3/agent5 |
| 2,3,7 | CNN strided | *parked at c3 baseline* | agent4 applies postproc only (marginal +0.0014); optional agent5 secondary | — |
| **9,11** | **ViT** | **agent3 (APRIL analytic) + agent5 (DPS/patch-embed)** | two DIFFERENT methods; agent4 picks the better per model | agent2 |
| ALL | — | **agent1** override + **agent4** integrate | retrieval upgrade where it passes gate; iLRG+postproc on everything | — |

**Per-agent status & re-scoping (relay to each running chat — see prompts the human is pasting):**
- **agent1 (Retrieval) — KEEP running, refine.** Temper scope: verbatim-all ~5%; realistic win is a
  CelebA-200 top-identities gallery on *some* slots. Chase **Fantômas** (arXiv 2210.10651). Membership
  test with 4 crops **+ horizontal flips**; gate any paste on >0.95 SSIM + zero subspace residual.
- **agent2 (Analytic-exact) — KEEP running, elevate.** You OWN MLP 1/4/5/8 AND CNN 6/10/12. Detection
  checklist + BOTH recovery paths (Fowl consecutive-bin differencing, Boenisch ratio) + intra-cluster
  differencing denoise; screen CNN fc1 for imprint. Keep layer-0 bias grads. (BSS is an *optional*
  fallback here only if differencing underperforms — do not spend a slot on it.)
- **agent3 — INTERRUPT & REPURPOSE to ViT-analytic (APRIL).** Drop BSS (demoted). New job: implement
  **APRIL** (arXiv 2112.14087, in `breaching`) on **model9** (patch8, learnable pos65:
  `∂L/∂input = ∂L/∂pos_emb` → recover patch tokens → left-multiply patch-embed pseudo-inverse → pixels)
  and **model11** (verify its PE is learnable first — if not, APRIL weakens). Namespace `vit_*.py`,
  `outputs_vit/`, `reports/vit/STATUS.md`. This is the primary ViT lever.
- **agent4 (Build/integrate) — KEEP running, add L0+L5.** Build `proxy_labels.py` (iLRG counts, all 12)
  → per-label dedup + √k duplicate-averaging + Hungarian-aware proxy. Add the L5 chain (align ±3–5px+flip
  → per-image blur → mean/std match → dup-average). Enforce the gate: beat the bank on NN-SSIM + 16×16
  coarse SSIM by >0.003–0.005 AND pass TV/grain guard before any submission. Sole owner of
  `submission_current_best.pt`.
- **agent5 — INTERRUPT & RE-SCOPE to ViT-generative + strided-CNN.** GIFD/DPS identity-synthesis at
  B=128 is dead as an endpoint. New job: (a) **DPS diffusion + patch-embed projection on ViT 9/11**
  (complements agent3's analytic route — agent4 picks the winner); (b) *secondary, only if ViT stalls*:
  on-manifold projection of blurry CNN 3/7 recon (must clear +0.003–0.005 vs bank). Keep `gp_*.py`,
  `outputs_gp/`.
- **agent6 (research, optional):** confirm Fantômas/CelebA-200 provenance; check whether the 200-way
  head is identical across all 12 (shared label space → cross-model membership); deep-dive APRIL
  applicability + whether model11's positional embedding is learnable.

---

## PART 0.6 — CONSOLIDATED TO 3 AGENTS + VERIFIED FINDINGS (this supersedes the 5-agent split)

All agents were stopped. Restart with THREE focused agents (A/B/C). Findings below were VERIFIED by
the main chat against the first-run artifacts on JURECA — treat as ground truth.

**Verified findings:**
- `outputs_retr/ground_truth.pt` is **NOT our ground truth.** It's a *different* SprintML face set
  (10 models, 150-class; `source_model` names like `cnn_tanh150_10ep_pool16`, `mlp_relu_50ep`).
  Cross-match vs our gradient subspaces: **0 exact members** (closest per-image residual 0.09; best set
  0.18 vs 0.50 random = same-distribution only). **Do NOT submit it.** Value = a same-distribution face
  gallery + proof the source is a public *labeled* face set.
- **Trap triage (`trap_diag.py`, all 12): ONLY model10 is trap** (Boenisch-sparse → near-verbatim via
  ratio + local differencing). All other 11 are honest (ratio = blend).
- **Noise structure** (E128 = energy in top-128 singular dirs): perturbed = MLP1/4/5 (0.77), MLP8
  (0.47), CNN6 (0.31); **CLEAN rank-128 (E128=1.0) = CNN 2,3,7,10,12 and ViT 9,11.** ⇒ the CNN `fc1`
  gradients of 2,3,7,10,12 are **exactly recoverable** — features are clean, only conv-inversion limits
  them. **model10 = trap + clean + stride-1 = the single best target.**
- Current c3 capture (proxy): MLP1/4/5 ≈0.73–0.75 (~96% of their ceiling → capture is tapped, gains now
  come from DENOISE), CNN2/3/7 ≈0.99, **model6 0.188 (worst, only 61% of its ceiling)**. **L5 postproc
  has NOT been applied — a reliable +0.05–0.15 is still unbanked.**

**THREE AGENTS — use chats agent1, agent2, agent3; retire agent4 & agent5** (their code stays on
JURECA and is reused). See the human's paste-ready prompts.
- **agent1 = Analytic reconstruction** — owns MLP 1,4,5,8 + CNN 2,3,6,7,10,12. Priorities: (1) model10
  trap near-verbatim; (2) clean CNN feature recovery (2,3,7,10,12) → best conv-invert; (3) intra-cluster
  differencing DENOISE on noisy MLPs + model6. Namespace `rec_*.py`, `outputs_trap/`.
- **agent2 = ViT → CLOSED, reassigned to RETRIEVAL.** ViT is a proven DEAD END: APRIL is a B=1 attack;
  first-layer grad eff-rank is only ~16 (m9)/~31 (m11) (the "E128=1.0" was a ≤256-singular-value
  artifact); self-test = B=1 SSIM 1.0 → B=128 best-per-sample 0.19/0.196, worse than the bank. **ViT 9/11
  stay on the diverse bank** (`outputs_vit/vit_{9,11}.pt`, 128 unique each). See `reports/vit/STATUS.md`
  + [[fl-recon-vit-april]]. Do NOT re-attempt gradient inversion on ViT. agent2 now owns **RETRIEVAL**
  (the only remaining ViT upside + biggest overall upside): find the real public source
  (CelebA top-200 / Fantômas 2210.10651), subspace-certified NN match, keep only residual≈0 exact hits.
  Namespace `retr_*.py`, `outputs_retr/`. Kill-switch: if the <1h membership test finds no verbatim
  match, pivot to helping agent1 on the weak models (model6 + strided CNN 2,3,7).
- **agent3 = Integrate + labels + postproc + retrieval-gate** — iLRG labels; L5 postproc applied to c3
  NOW for a fast safe submission; proxy + gate; sole owner of `submission_current_best.pt`. Bounded
  side-quest: download the REAL public source (CelebA-200 / Fantômas) and NN-match with the subspace
  certificate — keep only residual≈0 exact hits, abandon if none. Namespace `merge_*/proxy_*/retr_*`.

---

## PART I — Mission & the one number that matters

Reconstruct 128 private 64×64 RGB faces for each of 12 models (1536 total). Score =
per-model **Hungarian one-to-one SSIM**, averaged. We are at 0.33, leader at 0.48. The
gap is one or two *methods we have not tried*, not tuning of what we have.

**The metric's exploitable properties (from the official Task 3 PDF):**
- One-to-one matching → **coverage of distinct faces beats duplicating your best face.**
- Public leaderboard = 30% only; **final = hidden 70%** → no public-split overfitting.
- 5-min cooldown between successful submits (2-min on error). Leaderboard keeps *best-only*.
- **We are the "malicious server."** Models are given with **different defenses**: some are
  **trap-weighted (extra leaky)**, some are **noise-defended**. This is a per-model triage game.

---

## PART II — Why we are stuck, and the strategic reframe

Prior sessions concluded "0.415 is a near-ceiling" from a **subspace-capture** argument
(our 128 images span ~96% of the first-layer gradient's top-128 energy). **That argument
is wrong for SSIM.** Two independent reasons, both verified this session:

1. **Capture ≠ SSIM.** The first-layer weight gradient is `gW = Σ_b (dL/dz_b) x_bᵀ`
   (rank ≤ 128; the 128 images span its right-singular subspace). A neuron activated by
   *3 samples* yields a ratio-row `gW[j]/gb[j]` = a **blend of 3 faces** — it lives in the
   image subspace (so it counts toward "capture") but is **no real face** (low SSIM). Our
   whole pipeline is analytic-ratio + cluster + dedup; it never **separates the batch into
   the 128 individual faces**. Subspace *coverage* was mistaken for face *recovery*.

2. **Grain is the SSIM killer, and retrieval/denoise bypass the "noise ceiling" entirely.**
   The analytic ratio recovers **real but noisy** faces (≈40–50 recognizable faces/model
   under heavy RGB grain — visually confirmed). SSIM punishes grain far more than blur, and
   our c5 regression proved "more noisy faces" *lowers* score. The "defended gradient noise"
   caps *linear* reconstruction — but if the images are verbatim public data, we use the
   noisy recon only as a **query** to retrieve the clean original, and the gradient noise
   never touches the final image. The linear "ceiling" is irrelevant to that attack.

**So the path 0.33 → 0.48 is: turn noisy-but-recognizable faces into clean, pixel-faithful,
correctly-aligned faces** — via (a) retrieval of verbatim originals, (b) exact trap-weight
recovery, (c) true batch source-separation, (d) a strong generative prior, and (e) SSIM-aware
post-processing. Five levers below.

---

## PART III — Empirical findings from this session (do not re-derive)

- **Weight decay is NOT the extra-rank cause.** Best λ to cancel the W-component of `gW` is
  ~2e-8; subtracting λW leaves rank99≈959 and E@128≈0.768 (model1) unchanged. The extra rank
  is genuine additive/DP-style perturbation → **not removable by linear algebra.** (Probe:
  scratchpad `ica_probe.py`.)
- **Naive FastICA(128) on `gW.ᵀ` does NOT beat the ratio attack out of the box** — it
  under-converged into mean-face ghosts + ~1 crisp face. Real BSS needs the full recipe
  (whiten to top-128 subspace, negentropy + TV + decorrelation + sign/`nv` term, scale via
  bias gradient). Payoff real but needs engineering — see Lever 3.
- **Ratio grid = ~40–50 real distinct faces/model under grain.** Frontal, centered, aligned —
  **consistent with verbatim aligned-CelebA.** Strong prior for Lever 1.
- **Data source unconfirmed.** `hackathon_setup.sh` only pulls `SprintML/FL_Data_Reconstruction`
  (no dataset card returned). Provenance must be settled empirically (Lever 1, Agent R).
- **Cited papers = the intended toolbox:** Boenisch *trap weights* (2112.02918), Zhu *DLG*
  (1906.08935), Geiping *Inverting Gradients* (2003.14053). "Trap weights" is named in the
  task itself → **exact analytic recovery is in scope** (Lever 2).

---

## PART IV — The five levers (ranked by expected value)

> Every lever is stronger than what we've shipped, and all generalize to the hidden 70%.

**L1 — Public-dataset RETRIEVAL (verbatim originals).**  *Highest EV; test FIRST.*
If the 1536 images are verbatim CelebA / CelebA-HQ / FFHQ at a fixed 64px crop, use each
noisy recon as a **query** → nearest-neighbor retrieve the **exact clean original** → SSIM
per matched image → ~1.0. Under one-to-one matching, retrieving even 40–60% of a model's
batch lifts that model to ~0.6–0.7. **This alone is the most likely explanation for 0.48.**
Decisive 30-min test exists (see Agent R). Bypasses the gradient-noise ceiling entirely.

**L2 — Trap-weight EXACT recovery (Robbing-the-Fed differencing).**  *High EV on leaky models.*
The task is a *malicious-server* setting and names trap weights; some models are extra-leaky
(model10: 28% nnz + 261 dead-bias neurons = trap signature). For a bin/imprint structure,
**consecutive-bin differencing** `x = (gW_l − gW_{l+1}) ⊘ (gb_l − gb_{l+1})` cancels all other
samples → **verbatim** image for each non-colliding bin (Fowl 2021, arXiv 2110.13057). We only
ever did per-row ratio (averages), never differencing. Per-model triage for trap structure.

**L3 — True batch SOURCE SEPARATION (SPEAR++ / Cocktail-Party ICA).**  *Best pure-recon lever; generalizes.*
Our MLP first layer `Linear(12288→1024)`, B=128, is the textbook setting.
- **SPEAR++** (arXiv 2510.24200): sparsely-used dictionary learning — `argmin ‖L q‖₀` on the
  sphere (ℓ1/LogCosh + Riemannian Adam), **scale fixed by the bias gradient** (direct
  generalization of our ratio). Exact when **B ≲ hidden width** — width 1024, B=128 is inside
  the band (partial recovery shown to B=210).
- **Cocktail-Party Attack** (ICML'23, `facebookresearch/cocktail_party_attack`): ICA on the
  gradient rows, negentropy + TV + decorrelation + sign (`nv`) + L1; runs to B=1024, ships a
  `batch_size=128` MLP config. Ref hyperparams: `lr=1e-3, tv=3.1, T=12.4, decor=1.47, nv=0.13`.
  **Sign ambiguity** → pick the in-[0,1]/low-TV orientation; **permutation** is free under
  Hungarian. Seed `W_hat` from analytic-ratio rows for fast convergence. For CNN, run on the
  **fc1 input** (conv feature) then invert the conv.

**L4 — Generative PRIOR (GIFD feature-domain StyleGAN + DPS diffusion).**  *Rescues ViTs/strided CNN; big SSIM multiplier.*
Benchmarked SSIM on **FFHQ 64×64** (our exact setting): IG/DLG ≈ 0.20–0.23, GGL 0.245, GIAS
0.538, **GIFD 0.577**. GIFD (ICCV'23, `ffhibnese/GIFD_Gradient_Inversion_Attack`) does
progressive latent→W+→intermediate-feature optimization with a **per-stage ℓ1-ball projection**
— this projection is exactly the fix for our "high gradient-cosine but noise image" collapse.
**DPS diffusion** (`ffhq_10m.pt` + `DPS2022/diffusion-posterior-sampling`) as a plug-and-play
prior where the differentiable forward operator = "compute this model's averaged gradient" —
best for ViT 9/11 where StyleGAN failed. **No-CUDA path is solved:** NVlabs StyleGAN2 `torch_utils`
runs in pure-PyTorch `impl='ref'` (auto-fallback); DPS/BigGAN/ArcFace/GFPGAN are all pure PyTorch.
Reuse our r1/r2 no-plugin standup. Score SSIM **only after a ±3px alignment sweep** (generative
recons are shifted). Restoration (GFPGAN/CodeFormer) is demo-only: it hallucinates and can *lower*
SSIM — keep only if measured SSIM improves.

**L5 — SSIM-aware POST-PROCESSING (free, uniform, stacks on everything).**
Denoise + light Gaussian blur (σ≈0.5–0.8), **per-channel mean/std matching to the CelebA-64
distribution**, and a **±3px/small-scale alignment sweep** maximizing SSIM to a reference. SSIM
is low-pass-biased and shift-sensitive; grain is expensive, mild blur cheap/beneficial. Prefer
**diverse medium-quality** over few-excellent-many-bad (Hungarian). Optionally finish with direct
differentiable-SSIM ascent against a retrieved/denoised reference.

---

## PART V — Agent assignments (start these chats)

**Start 5 worker chats now** (the human's chats **agent1–agent5**), plus optional **agent6**
for continuous research. They are independent (own file-prefix + output dir → no collisions).

**Chat mapping & priority (watch in this order):**
`agent1 = R (retrieval)` · `agent2 = T (trap/exact)` · `agent3 = S (source-separation)` ·
`agent4 = B (build/measure — start early, runs continuously)` · `agent5 = G (generative prior)` ·
`agent6 = Research (optional, continuous)`. The **main chat** also runs a background deep-research
workflow that writes `task3/reconstruction/RESEARCH_FINDINGS.md` — every agent should read it once
it appears. Paste each brief verbatim into its chat; each is self-contained and points back here.

Namespaces (enforce to avoid git/FS collisions):
| chat | agent | code prefix | JURECA outputs | report file (owns) |
|---|---|---|---|---|
| agent1 | R (retrieval)   | `retr_*.py` | `recon_work/outputs_retr/`, `recon_work/gallery/` | `reports/retr/STATUS.md` |
| agent2 | T (trap/exact)  | `trap_*.py` | `recon_work/outputs_trap/` | `reports/trap/STATUS.md` |
| agent3 | S (BSS)         | `bss_*.py`  | `recon_work/outputs_bss/`  | `reports/bss/STATUS.md` |
| agent4 | B (build/merge) | `merge_*.py`,`proxy_*.py` | `recon_work/outputs_final/` | `reports/build/STATUS.md` |
| agent5 | G (gen prior)   | `gp_*.py`   | `recon_work/outputs_gp/`, `recon_work/runs/` | `reports/gp/STATUS.md` |
| agent6 | Research        | *(no code)* | *(none — writes docs)* | `reports/research/STATUS.md` |

Shared rules for ALL agents: read Part VII before touching JURECA; **never** edit the challenge
`models/`,`gradients/`,`.venv`; **never** overwrite `recon_work/outputs/submission.pt` or the
`_c3`/`_v1` artifacts; **never** handle the API key (the human submits); edit code **locally**,
sync + run on JURECA; write only under your namespace; commit only your own prefix files.

---

### ▶ CHAT R — Retrieval & data provenance (PRIORITY 1, do the 30-min binary test first)
```
You are Agent R on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md fully
(Parts II–VII). Your lever is L1 (verbatim public-dataset retrieval) — the highest-EV,
test-first workstream. Namespace: code `retr_*.py`, outputs recon_work/outputs_retr/ &
recon_work/gallery/, report reports/retr/STATUS.md. Do not touch other agents' files or the
challenge data/venv; the human submits (never touch the API key).

GOAL: determine, in <1h, whether the 128-images-per-model are verbatim images from a public
face dataset, and if so retrieve the exact clean originals at the correct 64px crop.

STEP 0 (provenance): inspect the HF dataset repo `SprintML/FL_Data_Reconstruction` (list files,
read any generation/metadata/scripts) and `SprintML/hackathon`. Look for the source dataset name
and preprocessing. JURECA login node HAS internet.
STEP 1 (galleries): download CelebA (`flwrlabs/celeba` or `nielsr/CelebA-faces`), CelebA-HQ, and
FFHQ. Build candidate 64px galleries under the standard crops: center-crop 178→64, 148→64,
140→64, 128→64, and HF-native 64. Each gallery = N×(3,64,64) in [0,1]. Isolated deps via
`uv pip install --target recon_work/runs/pylibs` (faiss-cpu, insightface or facenet-pytorch).
STEP 2 (DECISIVE membership test): take our best MLP recon (model1/4/5 from
recon_work/outputs_c3 or candidates). (a) pixel/SSIM NN into each gallery — is there a >0.95-SSIM
twin? (b) GRADIENT-SUBSPACE certificate: a true member lies in rowspace(gW); project the
candidate original onto the top-128 right-singular subspace of that model's net.0 weight-gradient
(reuse attack_mlp_svd_project.py) — a true member has ~0 residual, a random face does not. If a
gallery yields near-zero-residual, high-SSIM twins → the data is verbatim public and you've found
the crop. Report the answer as a clear YES/NO with evidence.
STEP 3 (if YES): retrieve for ALL models. Robust query = ArcFace/FaceNet embedding cosine (FAISS)
— robust to grain — but SUBMIT the correctly-cropped gallery tensor, not the ArcFace crop. Accept
a retrieval only when embedding + pixel + gradient-subspace-residual agree (few false positives).
Produce recon_work/outputs_retr/retrieved_{model}.pt (128 each). For unmatched slots, leave a flag
so Agent B fills from best reconstruction/bank.
STEP 4 (if NO / re-cropped): try FFHQ & LFW galleries; if still no member passes the subspace test,
declare L1 dead and pivot to helping Agent G. Report either way in reports/retr/STATUS.md.

Measurement w/o ground truth: gradient-subspace residual + re-feeding the retrieved batch to check
the produced gradient's cosine vs observed. Report per-model #verbatim-hits and mean residual.
```

### ▶ CHAT T — Trap-weight & exact analytic recovery (PRIORITY 2)
```
You are Agent T on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md fully.
Lever L2 (trap-weight exact recovery via Robbing-the-Fed differencing) + per-model defense triage.
Namespace: code `trap_*.py`, outputs recon_work/outputs_trap/, report reports/trap/STATUS.md.
Rules per AGENT_SETUP Part V.

GOAL: for each model, detect trap/imprint structure and, where present, recover VERBATIM images by
consecutive-bin differencing instead of the per-row ratio we've been using.

TASKS:
1. Per-model first-layer diagnostics (trap_diag.py): weight/grad histograms, bias-gradient ordering,
   fraction of dead/negative-bias neurons, activation sparsity, singular spectrum. Flag models whose
   first layer looks like an ordered bin/imprint stack (Boenisch trap weights / Robbing-the-Fed).
   model10 is the prime suspect (261 dead-bias neurons); check ALL 12.
2. For flagged models: implement Robbing-the-Fed exact recovery — sort neurons by bias (or by the
   cumulative-threshold structure), then x_t = (gW[l] − gW[l+1]) / (gb[l] − gb[l+1]). Each
   non-colliding consecutive pair yields one verbatim image. Handle collisions (a difference that is
   a blend) by rejection. Compare recovered images to our analytic-ratio faces.
3. Even for NON-trap MLPs: differencing neurons that share a single dominating sample denoises better
   than averaging — test it.
4. Verify each recovered image by re-feeding (with an inferred label from infer_labels.py) and
   checking its rank-1 contribution matches the observed gradient (near-zero subspace residual).
Deliver recon_work/outputs_trap/trap_{model}.pt (up to 128 each) + a per-model table of
#verbatim-recovered in reports/trap/STATUS.md. Refs: Fowl 2021 arXiv 2110.13057
(github.com/lhfowl/robbing_the_fed); Boenisch 2112.02918.
```

### ▶ CHAT S — Batch source separation: SPEAR++ / Cocktail-Party ICA (PRIORITY 3)
```
You are Agent S on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md fully.
Lever L3 (true batch source separation on honest MLP/CNN). Namespace: code `bss_*.py`, outputs
recon_work/outputs_bss/, report reports/bss/STATUS.md. Rules per AGENT_SETUP Part V. Note: the
naive FastICA probe (scratchpad ica_probe.py) UNDER-converged — you must implement the full recipe.

GOAL: separate each honest model's averaged first-layer gradient into the 128 INDIVIDUAL faces
(our ratio+cluster pipeline only recovers single-sample-dominant neurons and blends the rest).

TASKS:
1. Cocktail-Party Attack (bss_cpa.py) on MLP net.0 (models 1,4,5,8): whiten gW to the top-128
   subspace (PCA), learn a 128×128 orthogonal unmixing by Adam maximizing negentropy(log-cosh) with
   TV + decorrelation(exp(T·|cos|)) + sign/`nv`(min(‖relu(−s)‖,‖relu(s)‖)) + L1. Fix scale with the
   bias gradient; fix sign by the in-[0,1]/low-TV orientation. Seed unmixing from analytic-ratio rows.
   Hyperparams start: lr=1e-3, tv=3.1, T=12.4, decor=1.47, nv=0.13; sweep tv & T. Reference:
   github.com/facebookresearch/cocktail_party_attack (src/gradient_inversion.py).
2. SPEAR++ (bss_spear.py): SVD gW → L,R; solve argmin ‖L q‖₀ on the unit sphere via ℓ1/LogCosh +
   Riemannian-Adam (~1e5–1e6 inits, keep sparse-‖Lq‖ candidates, filter by sparsity-match λ); scale
   columns via the bias gradient (SPEAR++ Thm 4.1). Width 1024, B=128 is inside the recoverable band.
   Refs: arXiv 2403.03945 (SPEAR), 2510.24200 (SPEAR++).
3. CNN (6,10,12 and strided 2,3,7): run the same BSS on the fc1 gradient to separate 128 conv-feature
   maps, then invert the conv (reuse attack_conv_invert.py) to RGB.
Deliver recon_work/outputs_bss/bss_{model}.pt (aim 128 distinct each). Measure by gradient
self-consistency + subspace residual + visual grids (scp PNGs, Read them). Report which method wins
per model in reports/bss/STATUS.md. Heavy runs go through sbatch (see AGENT_SETUP Part VII).
```

### ▶ CHAT B — Build, measure, integrate, SSIM post-process (PRIORITY 4, runs continuously)
```
You are Agent B on CISPA Hackathon Task 3, the integration/measurement owner. Read
task3/reconstruction/AGENT_SETUP.md fully. Levers L5 + assembly. Namespace: code `merge_*.py`,
`proxy_*.py`, outputs recon_work/outputs_final/, report reports/build/STATUS.md. You OWN the
canonical recon_work/outputs_final/submission_current_best.pt. Rules per AGENT_SETUP Part V; the
human submits — never touch the API key.

GOAL: keep an always-valid best submission, measure candidates without ground truth, and squeeze
free SSIM via post-processing.

TASKS:
1. Local PROXY scorer (proxy_score.py) — we cannot see GT, so build the best available signals:
   (a) gradient self-consistency: re-feed a model's 128 imgs (with inferred labels) and report cosine
   vs observed gradient + top-128 subspace residual; (b) diversity/coverage stats; (c) grain/TV.
   Produce a per-model, per-source scoreboard so we choose sources rationally.
2. SSIM post-processing (merge_postproc.py): per-image mild denoise + light Gaussian blur (σ 0.5–0.8),
   per-channel mean/std matching to a CelebA-64 reference, optional ±3px/scale alignment sweep against
   the chosen reference. Verify it never increases grain. Make it toggleable per model.
3. Integrator (merge_build.py): per model, pick the best source among {c3 baseline, outputs_retr,
   outputs_trap, outputs_bss, outputs_gp} using the proxy scoreboard; enforce exactly 128 distinct,
   finite, [0,1], (128,3,64,64); fill shortfalls with diverse bank faces (never noise). Always leave a
   valid submission_current_best.pt + a human-readable manifest of which model came from which source.
4. Run validate_submission.py on every build. Maintain a submission LOG (score deltas the human
   reports back) so we spend the 5-min-cooldown submissions wisely — never burn one on an unverified
   change; batch improvements, submit deliberately, keep best-only.
Report the current scoreboard + recommended next submission in reports/build/STATUS.md.
```

### ▶ CHAT G — Generative prior (PRIORITY 5; rescues ViT 9/11 & strided CNN 2/3/7)
```
You are Agent G on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md fully.
Lever L4 (generative prior). Namespace: code `gp_*.py`, outputs recon_work/outputs_gp/ &
recon_work/runs/, report reports/gp/STATUS.md. Reuse the r1/r2 no-plugin StyleGAN standup
(gan_prior.py, attack_stylegan.py; see memory: NVlabs ops force impl='ref', isolated pylibs via
uv --target). Rules per AGENT_SETUP Part V.

GOAL: reconstruct the hard models (ViT 9,11; strided CNN 2,3,7; and refine anything blurry) with a
strong face prior — benchmark SSIM on FFHQ-64 is GIFD 0.577 vs our pixel-space ~0.2.

TASKS:
1. GIFD-style feature-domain optimization (gp_gifd.py) on StyleGAN2-FFHQ-64: progressive
   latent→W+→intermediate-feature stages with a PER-STAGE ℓ1-BALL PROJECTION (this is the fix for
   "high gradient-cosine but noise image") + TV(1e-4)+L2(1e-6). One latent/feature per sample, shared
   generator; infer labels; many restarts, keep best per-sample by gradient cosine. Force impl='ref'
   (no nvcc). Ref: ICCV'23 arXiv 2308.04699, github.com/ffhibnese/GIFD_Gradient_Inversion_Attack.
2. DPS diffusion prior (gp_dps.py) for ViT 9/11: plug-and-play with ffhq_10m.pt where the
   differentiable forward operator = "compute this model's averaged gradient (downsample 256→64)";
   add guidance −ζ∇‖y−A(x̂₀)‖ each denoise step; multi-restart + align-and-fuse to fight spatial
   jitter. Pure PyTorch (github.com/DPS2022/diffusion-posterior-sampling, openai/guided-diffusion).
3. ALWAYS align (±3px) before scoring SSIM. Treat GFPGAN/CodeFormer as demo-only (they can lower
   SSIM by hallucinating) — keep only if measured SSIM improves.
Deliver recon_work/outputs_gp/gp_{model}.pt for the hard models. If a model's prior output can't beat
the diverse bank on the proxy metric, say so and defer to bank. Report in reports/gp/STATUS.md.
```

### ▶ CHAT agent6 — Continuous research (OPTIONAL; complements the main chat's research workflow)
```
You are agent6 on CISPA Hackathon Task 3, the continuous researcher. Read
task3/reconstruction/AGENT_SETUP.md fully, and FIRST read task3/reconstruction/RESEARCH_FINDINGS.md
if it exists (the main chat's background research writes it). Your job is to go BEYOND that doc — do
NOT repeat findings already there. You do not run code or touch JURECA; you produce written
intelligence in reports/research/STATUS.md and flag anything decision-changing to the main chat.

STANDING TASKS (iterate over the 12h, following live leads from the other agents):
1. DATA PROVENANCE (top priority): pin down the EXACT public source dataset + 64px preprocessing for a
   200-class 64x64 face set. Search HuggingFace SprintML org, CISPA/SprintML papers (Boenisch/Dziedzic),
   any leaked challenge details, GitHub for FL_Data_Reconstruction generation scripts. If agent1 reports
   its membership test, deep-dive whichever crop/dataset is closest.
2. Whichever lever agent1–agent5 report as LIVE, find the best published implementation tricks + failure
   fixes for it (e.g., if BSS is live, dig CPA/SPEAR++ hyperparameter tuning; if prior is live, dig
   GIFD/DPS 64px gotchas). Keep a running "open questions the execution agents are blocked on" list and
   answer them.
3. Hunt for NOVEL angles not in the 5-lever plan: secure-agg breaks, ViT patch-embedding analytic
   (APRIL/GradViT), diffusion-prior batch handling, SSIM metric exploits, label-multiset recovery for
   200-class softmax (iLRG). Verify against primary sources; state confidence.
Report concise, citation-backed updates in reports/research/STATUS.md; escalate decision-changing finds
to the main chat immediately. Do not duplicate the main chat's RESEARCH_FINDINGS.md.
```

---

## PART VI — Shared protocol & submission discipline

- **Measurement without ground truth** (standardize on these — Agent B owns the harness):
  1) **gradient self-consistency** — re-feed a model's 128 images (with inferred labels) and compare
     the produced gradient to the observed one (cosine); 2) **subspace residual** — a true image lies
     in `rowspace(gW)`; low residual = plausible; 3) **diversity/coverage** and **grain/TV**. These
     are proxies; the ground truth is the leaderboard, which is precious (5-min cooldown, 30% only).
- **Submission discipline:** the human submits. Keep `outputs_final/submission_current_best.pt` always
  valid. Batch improvements; submit deliberately; never burn a cooldown on an unverified change; log
  every reported score delta. Optimize for the *hidden 70%* — prefer methods that generalize
  (retrieval, exact recovery, BSS, prior) over anything that looks like public-split fitting.
- **No file collisions:** each agent writes only its namespace (Part V). Main owns this file. Each
  agent owns its `reports/<agent>/STATUS.md`. Do not edit another agent's files.
- **Escalate to main** (this chat) if: a lever is confirmed dead, two levers want the same slot, or a
  cross-cutting decision is needed (e.g., "L1 is live → everyone pivot to retrieval + alignment").

---

## PART VII — Ops cheat-sheet (every agent needs this)

**Topology.** Windows 11 laptop; 3 shells + 1 cluster. `Bash` tool = **Git Bash** (mangles unix
paths — always wrap `wsl.exe bash -c '...'`). **WSL (Ubuntu)** holds the JURECA SSH key/config/socket.
**JURECA** = FZ-Jülich Slurm cluster (A100). **Golden rule: everything JURECA goes through WSL.**

**Reach JURECA (SSH + TOTP + multiplexing).** Login needs SSH key + TOTP. The human runs `ssh jureca`
once per ~8h in a WSL terminal (types TOTP) to prime a ControlMaster socket; scripted calls then reuse
it with no prompt. Test: `wsl.exe bash -c 'ssh -o BatchMode=yes jureca hostname'` → prints a hostname.
If `Permission denied (keyboard-interactive)` → socket expired, **ask the human for TOTP**; never try
to satisfy TOTP yourself. GitHub from JURECA is via `ssh.github.com:443` (deploy key); clone URL
`git@github.com:pwnXtoXe4/CISPA-Hackathon.git`. **Login node HAS internet** (HF/pip downloads OK);
**compute nodes are offline** (pre-download weights/data on login node first).

**Paths.**
| what | path |
|---|---|
| Local repo (Win) | `C:\Users\flori\OneDrive\Dokumente\hackathon\cispaFinale\CISPA-Hackathon` |
| Local repo (WSL) | `/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon` |
| Pipeline code | `task3/reconstruction/` |
| JURECA team root | `/p/scratch/training2625/zainzinger1/1337` |
| **Challenge data (read-only!)** | `…/1337/FL_Data_Reconstruction/{models,gradients}/model{1..12}.pt` |
| venv | `…/1337/FL_Data_Reconstruction/.venv` (`source …/bin/activate`) |
| **Workspace (all outputs)** | `…/1337/recon_work/` (`code/ reports/ candidates*/ outputs*/ runs/ output/`) |
| Synced code | `…/1337/recon_work/code/` (scp target; run from here) |
| Slurm logs | `…/1337/recon_work/output/<jobid>.{out,err}` |

Slurm: account `training2625`, reservation `cispahack`, partition `dc-gpu`, 1×A100-40GB.
venv: Python 3.12, `torch 2.12.1+cu130`, numpy, scikit-image (SSIM), scikit-learn (**FastICA present**),
scipy, PIL. **NO torchvision / lpips / matplotlib.** Extra deps → isolated
`uv pip install --target recon_work/runs/pylibs <pkg>` then `sys.path.insert(0,'…/pylibs')`.

**Dev loop.** Edit code **locally** in `task3/reconstruction/` → sync + run on JURECA. A `sync_exec.sh`
in the session scratchpad scps `*.py`+`*.sh` to `recon_work/code/`, strips CRLF, and runs a command in
the venv with `FL_WORK_ROOT`/`FL_DATA_ROOT` set. Pattern (single file):
`wsl.exe bash -c 'scp -q -o BatchMode=yes "/mnt/c/…/foo.py" jureca:…/recon_work/code/ && ssh -o BatchMode=yes jureca "cd …/recon_work/code && source …/.venv/bin/activate && export FL_WORK_ROOT=…/recon_work FL_DATA_ROOT=…/FL_Data_Reconstruction FL_DEVICE=cpu && python -u foo.py"'`
Code is env-driven: `FL_DATA_ROOT`, `FL_WORK_ROOT`, `FL_DEVICE` (cpu/cuda). **View images:** `scp` the
PNG to local scratchpad, then `Read` it (renders inline). Generate grids with `utils.save_image_grid`.

**GPU jobs (sbatch).** Any optimization / inversion / ICA-heavy / prior run goes through Slurm, NOT the
login node. SBATCH header:
```
#SBATCH --account=training2625 --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
```
Submit with env overrides via `--export=ALL,FL_ITERS=...`. Monitor `squeue -u zainzinger1`; history
`sacct -X --starttime today --format=JobID,JobName,State,Elapsed`. Light analytic math / SVD / a few
thousand SSIM calls are login-node-safe if short and single; **inversions/optimization/ICA are not.**

**Traps that already cost time.** (1) Inline `$VAR`/`$(...)` through `wsl.exe bash -c '…'` gets eaten —
put anything with shell vars in a **script file**. (2) Nested SSH quoting breaks on parens/redirects —
pipe a local `.sh`: `ssh jureca "bash -s" < script.sh`. (3) Git Bash path mangling — always wrap
`wsl.exe bash -c 'bash "/mnt/c/…/x.sh"'`. (4) CRLF on `.sh` breaks sbatch — `sed -i 's/\r$//' *.sh`.
(5) Don't stack CPU jobs on the login node. (6) `attack_conv_invert.invert_conv` is unseeded →
CNN inversions differ per run. (7) ViT gradient-matching reaches cosine ~0.9 but yields color-noise —
don't chase it; use prior (L4) or bank.

**House rules.** Edit locally, never on JURECA (scp overwrites). Never modify/commit challenge
`models/`,`gradients/`,`.venv`,`data/`,`output(s)/`, caches, secrets. Never overwrite
`recon_work/outputs/submission.pt`, `submission_v1_bank.pt`, or the `_c3` artifacts. **Never handle the
API key / HF / GitHub tokens / TOTP** — the human submits. Ask before `rm -rf`/`git reset --hard`.
Plain `git add/commit/push` is safe (triggers no job); only `deploy_submit.sh` triggers a JURECA pull+sbatch.
