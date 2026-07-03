# Task 3 — Deep Research Findings (novel approaches)

> Produced by the main chat's adversarially-verified research workflow (6 finders → 3
> refutation verifiers → synthesis, all Opus, ~673k tokens, primary sources only).
> **Every execution agent (agent1–agent6) reads this once.** It REVISES the 5-lever plan in
> AGENT_SETUP.md — see the re-scoping there.

## 1. TL;DR — three highest-leverage NEW insights + plan changes

1. **iLRG label-MULTISET recovery is free, universal, near-zero risk (NEW → lever L0).** All
   prior work recovered only the label *set*; iLRG (ICLR'23, OpenReview `FIrQfNSOoTr`, code
   `BUAA-CST/iLRG` on the **`master`** branch — `main` 404s) solves a `(C+1)×C` pseudoinverse for
   the exact per-class **counts** `k_j` at B≤4096. It (a) splits the 128×128 Hungarian into
   per-label blocks, (b) tells you the number of *distinct* identities so you cluster+average
   duplicates for a √k denoise, (c) gives class-conditional priors. Runs on the 200-way head of
   all 12 models. **ADD as L0, feeding L1/L3/L4/L5.**

2. **Differencing is the concrete, primary-source mechanism L2 was missing.** Two *distinct*
   named attacks: **Boenisch trap weights** (`z₊=−s·z₋`; per-neuron ratio `x=gW_i/gb_i`, Eq. 6)
   vs **Fowl imprint/binning** (duplicated weight rows + monotone-CDF biases → consecutive-bin
   differencing). A **detection checklist** (`dup_frac`, `w_rank1`, `bias_monotone`,
   `frac_neg≈0.5`, `dead_bias`) routes each model to the right formula, and **intra-cluster
   differencing** `(gW_i−gW_j)/(gb_i−gb_j)` cancels shared DP/other-sample noise — directly
   attacking the "grain is the SSIM-killer" problem. Expect **60–100/128 verbatim** on flagged
   trap models. **L2 is now the highest-ROI lever after L0.**

3. **BSS and generative priors are BOTH refuted as standalone reconstructors at B=128 — demote.**
   (a) CPA breaks at DP σ=1e-4 (LPIPS 0.008→0.474), smaller than our measured perturbation
   (E@128≈0.77 ⇒ 23–53% off-subspace energy); SPEAR++ hits a cliff at exactly our regime
   (m=1000: b=100 works, **b=150→0% accuracy**) and dies on GELU model8. (b) GIFD's headline
   0.577 SSIM is a **B≤4** number (PSNR 21→12.7 by B=8); our own StyleGAN-latent already measured
   **−0.0093** on model2/9 and lost to random on model11. **Plan change: L3 (BSS) survives ONLY
   as a source-quality improver feeding L1/L4 queries — mandatory rank-128 projection first,
   models 1/4/5 only. L4 (prior) survives ONLY as on-manifold projection/denoise of already-
   recovered CNN recons (+0.0014 measured), never identity synthesis. Do NOT budget leaderboard
   slots on either as endpoints.**

**Revised lever stack:** **L0 iLRG labels (new)** → **L2 trap-differencing (primary)** →
**L1 retrieval (test first, don't assume)** → **L5 postproc (align+blur+dup-avg, reliable)** →
**L3 BSS (source-only)** → **L4 prior (projection-only)**.

---

## 2. DATA PROVENANCE

**Verdict:** source is intentionally undisclosed. "Verbatim retrieval → SSIM 1.0 across all
1536" is **REFUTED (~5%)**; the actionable "some models draw from a CelebA-200 gallery, retrieval
lifts *those* slots" is **~15–20%**, worth a <1h test but **must NOT be budgeted on**.

- **Repo discloses nothing:** `SprintML/FL_Data_Reconstruction` has no dataset card / README /
  generation script; `requirements.txt` = torch/requests/numpy/scikit-learn/scikit-image (no
  torchvision/datasets — pipeline deliberately withheld). Task PDF names no dataset.
- **200-class fingerprint:** CelebA has 10,177 identities, so 200-way is a **curated top-N-
  identities subset**. Strongest match: the **Fantômas** anonymization paper (arXiv 2210.10651,
  CISPA/SprintML orbit) sorts CelebA by images-per-identity and keeps **exactly 200** (≈27 imgs/id
  ≈ 5,400 images — enough for 12×128). NOTE: the trap-weight paper (2112.02918) uses
  ImageNet/CIFAR/MNIST — the "200" link is **Fantômas, not Boenisch**.

**Gallery-build transforms (try in this order): CenterCrop 148 → 128 → 178 → plain Resize(64),
all `ToTensor()` only ([0,1], no normalize).**
```python
from collections import Counter
from torchvision import transforms as T
ids = [l.split() for l in open("identity_CelebA.txt")]        # "img.jpg  id"
top200 = {i for i,_ in Counter(int(i) for _,i in ids).most_common(200)}
files  = [f for f,i in ids if int(i) in top200]
variants = {
  "cc148": T.Compose([T.CenterCrop(148), T.Resize(64), T.ToTensor()]),  # DDPM default
  "cc128": T.Compose([T.CenterCrop(128), T.Resize(64), T.ToTensor()]),  # DCGAN default
  "cc178": T.Compose([T.CenterCrop(178), T.Resize(64), T.ToTensor()]),
  "r64":   T.Compose([T.Resize((64,64)), T.ToTensor()]),
}
```
Downloads: MMLAB CelebA page; Kaggle `jessicali9530/celeba-dataset`; HF `flwrlabs/celeba`,
`eurecom-ds/celeba` (178×178 + `celeb_id`). Alternates: CelebA-HQ identity
(`ndb796/CelebA-HQ-Face-Identity-...`, plain Resize(64) — HQ pre-aligned); LEAF native 64×64
(reuse *preprocessing* only).

**DECISIVE verbatim-membership test (analytic MLPs 1/4/5, <1h, NO leaderboard call):**
1. Best analytic recon `x̂` per sample (reuse `attack_mlp_svd_project.py`/`attack_analytic.py`).
2. Build the 4 crop galleries over top-200 files **and their horizontal flips**.
3. NN SSIM/LPIPS of each `x̂` vs gallery+flip. **Read the histogram:**
   - **Spike >0.95 for one crop** ⇒ verbatim, no aug → paste originals, SSIM→1 on matched slots.
   - **Plateau 0.5–0.8, flip often wins** ⇒ right dataset but **augmentation ON** → verbatim dead;
     retrieval usable as identity prior only.
   - **No crop beats the within-set nearest-different-face baseline** (`analyze_overlap.py::
     within_baseline`) ⇒ synthetic/bespoke → route to prior.
4. **Kill look-alike false positives:** project accepted candidate onto top-128 right-singular
   subspace of `net.0` weight-grad; residual≈0 for a true member (validates vs the *gradient*).

**Two kill-shots:** (a) scored GT is the **exact training tensor forward-passed** — a wrong-but-
plausible crop caps SSIM 0.4–0.7, not 1.0; (b) standard classifier augmentation (RandomResizedCrop
/HorizontalFlip/ColorJitter) makes verbatim SSIM~1.0 *mathematically impossible*, and the hidden
70% split exists to punish lookup. `P(deterministic, no-aug) ≈ 0.35`. Our clean-analytic MLPs
capping at 0.3325 (not ~1.0) is direct evidence verbatim-paste doesn't trivially win. **Run the
test, don't assume.**

---

## 3. Per-lever recipes + adversarial verdicts

### L0 — iLRG label MULTISET recovery (NEW; run FIRST on all 12)
Needs only head weight-grad `∇W∈R^{C×m}` + bias-grad `∇b∈R^C`, C=200, K=128.
1. Per-class avg embedding `e_i ≈ ∇W_i/∇b_i` (log-space, clamp exp ±10; division removes unknown
   scale γ — matters for leaky/trap models).
2. `p_i = softmax(α·(W·e_i+b))` via the white-box head; sweep **α∈{0.003,0.01,0.03}**, pick the α
   whose recovered counts sum closest to 128 (free self-consistency check).
3. Solve the `(C+1)×C` system by pseudoinverse:
```python
def ilrg_counts(probs, grad_b, C=200, K=128):
    coefs, vals = [[1]*C], [K]                 # closure Σk_j=K
    for i in range(C):
        row = [probs[j][i] for j in range(C)]; row[i] -= 1.0
        coefs.append(row); vals.append(K*grad_b[i])
    raw = np.linalg.pinv(np.array(coefs)).dot(np.array(vals))
    k = np.round(np.clip(raw, 0, K)).astype(int)
    d = int(k.sum()-K)                          # integer fixup to sum==K
    err = k - raw
    if d>0: k[np.argsort(-err)[:d]] -= 1
    if d<0: k[np.argsort(err)[:-d]] += 1
    return k, np.where(k>0)[0]
```
Use `sim_iLRG` (restrict system to the existence-set from `gi_infer` column-min<0 / `idlg_infer`
column-sum<0) for conditioning. **Verdict:** iLRG reports >99% LeAcc / >96% LnAcc to B=4096;
B=128@C=200 is comfortably inside. Untrained/lightly-trained ⇒ best; trap/leaky models sharpen ∇b
(helps). *Kills it:* DP-noise on the head → fall back to existence-only + uniform-count prior.
No-GT measure: α-sweep count-sum-vs-128. **Downside: none — run it.**

### L1 — Retrieval  (see §2). Conditional coin-flip. Do the membership test BEFORE writing any paste.

### L2 — Trap-differencing (PRIMARY after L0)
**Detection** (`trap_diag.py`, run on the first-biased Linear AND every Linear — imprint can be a
hidden inserted layer):

| Signal | Flag | → |
|---|---|---|
| `dup_frac>0.5` **or** `w_rank1>0.9` | duplicate rows | **Fowl imprint** → 3a differencing |
| `bias_monotone==True` | sorted CDF biases | imprint cumulative confirmed |
| `frac_neg∈[0.45,0.55]` + `dead_bias>0.2`, rows NOT dup | — | **Boenisch trap** → 3b ratio + 3c intra-cluster diff |
| `eff_rank≈128`, sharp `gap_128` (s[127]≫s[128]) | — | clean separable → analytic viable |
| `frac_neg≈0.5` random, `dead_bias≈0`, eff_rank noisy | — | honest/noise-defended → BSS/prior/retrieval |

**3a Imprint differencing** (breaching `analytic_attack.py`): sort rows by bias; if cumulative,
`for i in reversed(range(1,m)): gW[i]-=gW[i-1]; gb[i]-=gb[i-1]`; then `valid=gb!=0;
imgs=gW[valid]/gb[valid,None]` → each row `(3,64,64)` = verbatim single-sample bin.
**3b Boenisch ratio** (Eq. 6): `x=gW_i/gb_i`, valid iff `gb_i≠0` and exactly one sample fires i.
**3c Intra-cluster denoise (our MLPs 1,4,5,8):** within each ratio-cluster (reuse
`attack_mlp_exact_v2.py`), replace the trimmed-mean rep with the **best pairwise difference**
(largest `|gb_i−gb_j|` for stability) — cancels shared additive noise, sharpens grainy→verbatim.
**3d Collision rejection:** reject if `|gb|<1e-7`, `>15%` pixels outside `[-0.05,1.05]`, `TV>0.30`,
`std<0.05`, or cluster-consistency<0.5. Final: re-feed accepted image+inferred label, check cosine
+ top-B subspace residual≈0 (rank-1 certificate).
**Verdict:** highest-ROI after L0; Boenisch honest result >50% verbatim @ B=100; realistic
**60–100/128 on flagged models**. *Kills it:* layer-0 not followed by ReLU/trap → dense `∂L/∂Z`
(detect via near-zero sparsity fraction); collisions → blends. **Keep layer-0 bias gradient**
(needed for scale). Screen CNN `fc1` too (imprint there → conv *features* → `attack_conv_invert.py`).

### L3 — BSS (DEMOTED to source-quality only; models 1/4/5)
**CPA** (`facebookresearch/cocktail_party_attack`, arXiv **2209.05578** — not 2403.03945): attack
`grads[0]` `[1024,12288]`; whiten to 128 comps; optimize `W_hat` (128×128, eye init, Adam lr=1e-3);
FC config **tv=3.1, decor=1.47, T=12.4, 25k iters**; loss `= decor·(exp(T·|cos|)−1).mean() + tv·TV`;
per-source min–max→[0,1]. The `exp(T|cos|)` decorrelation is why the custom loop beats FastICA
(which under-converged to mean-face ghosts — matches our probe). **SPEAR++** (arXiv 2510.24200):
SVD `gW=LR`, recover each column of Q by ℓ1-min on the sphere (RiemannianAdam, 500 it, lr
1e-1→1e-3@200→1e-5@400), scale via bias-grad (Thm 4.1: `s=Q̄⁻¹L†∇b`), reconstruct `Xᵀ=Q⁻¹R`, keep
λ=1 columns. **MANDATORY pre-step:** rank-128 SVD truncation of gW (`subspace_denoise.py` /
`attack_mlp_svd_project.py`) — else both ingest 23–53% noise and diverge.
**Verdict (REFUTED as standalone):** CPA broken by DP σ=1e-4 (< our perturbation); SPEAR++ cliff at
m=1024/b=128 (b=150→0% even noiseless); dies on GELU model8; our ratio already captures 96% of the
E@128 ceiling. **Surviving use:** on 1/4/5 only, BSS may separate individual faces that ratio blends
→ feed cleaner sources as **retrieval/prior queries**, not endpoints. No-GT: count non-ghost faces +
gradient self-consistency; try ±S, keep lower-TV sign. **Never spend a submission slot on
BSS-as-reconstructor** (c5's extra noisy faces lowered the score).

### L4 — Generative prior (DEMOTED to projection/denoise only; CNN 3/7)
**GIFD** (arXiv 2308.04699): rosinality StyleGAN2 FFHQ-256 (`gdown 1JCBiKY_yUixTa6F1eflABL88T4cii2GR`),
progressive W+→noise→features PGD in ℓ1-balls, K=4 stop, Adam lr0.1, 1000it/stage, 4 restarts,
tv=1e-4 l2=1e-6, cosine cost, downsample 256→64 *inside* the operator. **impl='ref' gotcha:**
rosinality `op/fused_act.py`+`op/upfirdn2d.py` fall back to native PyTorch **only on CPU** — no GPU
ref path; patch `if input.device.type=="cpu"` → unconditional native and delete the `load()` JIT
import. Replace its unique-label restorer with iLRG (L0). **DPS** (`DPS2022/diffusion-posterior-
sampling`): guided-diffusion `ffhq_10m.pt` (DPS Google Drive, NOT OpenAI's ImageNet ckpt) — **pure
PyTorch, nothing to patch**. Operator `A(x)=avg_gradient(model, downsample→64(x), labels)` with
`create_graph=True`; conditioning `scale`=ζ (start 0.3), `timestep_respacing` 100–250, micro-batch
8–32 on A100-40GB. DPS better for ViT/noise-defended.
**Verdict (REFUTED as identity synthesis):** GIFD 0.577 is B≤4 (PSNR 21→12.7 by B=8); our
StyleGAN-latent already scored −0.0093; ViTs leak ~8 color modes (bank provably optimal); strided
CNN 2/3/7 are rank-starved (eff-rank 32/57/84<128 — prior can't invent missing dims). **Surviving
use:** on-manifold **projection** of already-recovered blurry CNN 3/7 recon = **+0.0014 measured**.
No-GT gate: must beat the diverse bank on `analyze_overlap.py` NN-SSIM + coarse-16×16-SSIM by more
than the c4/c5 noise floor (**0.003–0.005**) and pass the TV/grain guard before earning a submission.
±3px alignment before scoring; reconstruct at 256, downsample-in-operator (no 64 ckpt exists).

### L5 — Postproc (reliable, do always)
SSIM = `l·c·s`, 11×11 Gaussian σ=1.5. Prioritized (Δ per 64×64 face):

| # | Op | term | Δ |
|---|---|---|---|
|1| **Sub-pixel align** ±3–5px + flip, keep max-SSIM | s | **+0.05–0.15** |
|2| **Mild Gaussian blur** σ∈{0.5,0.8,1.0,1.3}, per-image best-vs-unblurred | c,s | +0.03–0.10 noisy / ~0 clean |
|3| **Per-channel mean+std match** to face ref (per-class mean if label known) | l,c | +0.02–0.06 |
|4| **TV denoise** (edge-preserving; prefer over blur when real edges) | s,c | +0.02–0.06 |
|5| **Cluster+average duplicates** via iLRG counts (√k denoise; submit k copies for a count-k class) | s,c | +0.03–0.08 on dup slots |

Never over-blur past σ≈1.5. MS-SSIM-loss polish only against a *retrieved* reference,
`normalize="relu"`, as the final step. **Diversity (Hungarian one-to-one):** filler faces score
~0.15–0.35 (NOT free); "128 uniform ~0.42" (avg 0.420) beats "60@0.60 + 68@0.20" (avg 0.388)
**unless** the excellent tier >0.7 AND fillers would be pure noise (~0). ⇒ trap-leaky models →
uniform quality; noise-defended → a bimodal split can win. iLRG-optimal: one good face per distinct
identity, duplicated across its k slots.

---

## 4. New methods worth ADDING (not in the original plan)

- **iLRG multiset (L0)** — the biggest genuinely-new addition (above).
- **APRIL for ViTs** (arXiv 2112.14087, in `breaching`): ViT-specific *analytic* leak — with a
  **learnable positional embedding**, `∂L/∂input = ∂L/∂pos-emb`; recover patch features then
  left-multiply the patch-embed pseudo-inverse → pixels. **Model 9 (patch8, learnable pos65) is the
  prime target; verify model 11's PE is learnable** (global-pool; if PE non-learnable, APRIL
  weakens — that's its own stated defense). Exact at b=1; combine with per-sample isolation. This is
  the analytic ViT route the plan lacked (GIFD/DPS on ViT are refuted). **Give this to agent5 or
  agent2.**
- **Cutting Through Privacy** (arXiv 2505.10264): malicious hyperplane isolation, distribution-
  agnostic, 42.5% @ B=4096 vs CAH-trap 0.37%; closed-form `x=(∂L/∂W)(∂L/∂b)⁻¹`. Only if we may
  inject/overwrite FC params — **verify scope** (challenge gives gradients from already-trained
  models; we likely CANNOT inject). Treat as a read-only template the trap models may already embed.
- **LOKI** (arXiv 2303.12233): conv imprint, 76–86% of 128 @ 64×64 Tiny-ImageNet — same inject
  caveat. Check CNN first layers.
- **SPEAR** (`eth-sri/SPEAR`): exact b≤25 core — use only on isolated ≤25-row sub-batches.

**Malicious-scope gate:** Cutting-Through / LOKI / Fishing all require *modifying sent weights*. We
get gradients from already-trained models → likely cannot inject. Treat as read-only closed-form
templates; L2 detection tells you if the trap models already embed such structure.

---

## 5. Ranked next-actions (next 12h)

1. **Read head width from `models/model*.pt`** — confirm out_features==200 and whether identical
   across all 12 (shared label space ⇒ cross-model membership). Confirm layer-0 `[1024,12288]`. ~5min.
2. **Run iLRG (L0) on all 12** — free labels + distinct-identity counts for every downstream lever;
   wire into Hungarian (per-label blocks) and L5 (dup-averaging).
3. **Run `trap_diag.py` (L2 detection) on all 12** — the flag table routes each model. Confirm
   model10; screen MLPs 1/4/5/8 + CNN `fc1`. Keep layer-0 bias grads.
4. **Run the verbatim-membership test (L1)** on analytic MLPs 1/4/5 — the <1h experiment that jointly
   settles dataset+crop+augmentation. Gate any retrieval-paste on a >0.95 spike.
5. **Apply L2 differencing** to flagged models (3a imprint / 3b+3c Boenisch) → `outputs_trap/`; gate
   through 3d + subspace-residual. Target 60–100 verbatim/model.
6. **Apply L5 postproc** (align→TV/blur→mean-std→dup-avg) to *all* current outputs — reliable
   +0.05–0.15 with zero new information. Do this even before new attacks land.
7. **BSS (L3) on 1/4/5 ONLY**, rank-128 projection first, as a *query generator* for L1/L4 — measure
   clean-face count, not aggregate capture. Skip model8 (GELU).
8. **APRIL on ViT 9** (and check PE-learnable on 11) — the analytic ViT route; if it beats bank on
   the proxy, it's new signal.
9. **Prior (L4) as projection-only** on CNN 3/7 recon — must clear the 0.003–0.005 proxy floor vs
   bank before a submission slot. Do NOT submit a GIFD/DPS latent-match swap on faith.

**Submission discipline (5-min cooldown):** every candidate must beat the diverse bank on the
GT-blind proxy (`analyze_overlap.py` NN-SSIM + 16×16 coarse SSIM) by more than the 0.003–0.005
model-level noise floor and pass the TV/grain guard before a leaderboard call. High gradient-cosine
is necessary-not-sufficient (v3 hit 0.98 cosine on pure noise) — use it only to *reject*.
