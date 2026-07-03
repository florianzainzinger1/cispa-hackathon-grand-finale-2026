# Task 3 — Approaches Ledger (single source of truth)

**Purpose:** before proposing ANY approach, grep this file. If it's under DEAD or
TAPPED, it's been tried and refuted **on the real leaderboard** — don't re-propose
without a concretely different mechanism. Every score here is **board-confirmed**
(a real response from `POST http://<SUBMIT_SERVER>/submit/21-fl-audit`, with the
server's own `delta`/`improved` flag) unless marked *(local/unverified)*.

- **Current best (board-confirmed): `test_pib2.pt` = 0.3888.** Promoted to
  `recon_work/outputs_final/submission_current_best.pt`.
- **Leader: ~0.48.** The gap lives in **base recon of the leaky models**, not post-proc.
- **Rate limit:** 1 submission / 5 min. Shared account — coordinate.

---

## The five laws (paid for in submissions)

1. **Only the leaderboard is truth.** Local proxies (cosine, subspace residual,
   capture-%, faithfulness, and the `ground_truth.pt` local `ssim_grader`) do NOT
   track the board. The `ff` episode: faithfulness went *up* on all 4 MLPs, board
   went **−0.0703**. Treat any non-board number as a hypothesis.
2. **Smooth beats faithful.** Adding sharp/faithful high-freq detail regresses. Grain
   is the SSIM killer.
3. **Diversity beats shrink — under Hungarian one-to-one.** Collapsing many slots to
   one smooth/mean face loses matches. (Killed the pooled-CNN inverter.)
   Tension: for *zero-signal* models, mean-face shrink is also SSIM-safe — decide by board.
4. **One variable per submission, never bundle.** Otherwise attribution is impossible.
5. **Triage math.** Leader ≈ a few models at ~0.7 + the rest at ~0.33. A FEW well-recon
   models move the average; marginal gains spread over all 12 do not.

---

## Verified architecture map (do not re-derive)

| Models | Family | Readout | Notes |
|---|---|---|---|
| **1, 4, 5, 8** | MLP | `net.0` input == image (in_dim 12288) | leaky, cleanest analytic; **headroom** |
| **10, 12** | CNN full-res | `head`/fc → invert conv (over-determined, clean) | leaky; **headroom** |
| **2, 3, 7** | CNN pooled | `fc1` → pooled map (k=4/8) | info-bottleneck; m2 readout degenerate (clean=0) |
| **6** | defended | — | DP-style noise, little signal |
| **9, 11** | ViT | — | no clean per-image readout; effectively dead |

All 12 are **200-class classifiers with NO BatchNorm** (checked). No BN ⇒ BN-dependent
attacks (GIRG, GradInversion, GradViT) lose their main side-channel.

---

## Board-confirmed scoreboard (chronological climb)

| file | score | Δ | lever |
|---|---|---|---|
| c3 (frozen analytic) | 0.3325 | — | analytic base |
| best_base (+model2 pick-3) | 0.3413 | +0.0088 | better m2 slots |
| **test_nlm05** | **0.3722** | +0.031 | **metric-level: blur σ1.0 + NLM h0.05** ← biggest single lever |
| test_vithi45 | 0.3837 | +0.0008 | ViT 9/11 mean-blend β0.45 |
| test_lowsig | 0.3851 | +0.0014 | ViT β0.55 + m3 β0.45 + m6 β0.40 |
| test_pib1 | 0.3878 | +0.0026 | per-image blend, full quality signal, γ0.10 |
| **test_pib2** | **0.3888** | +0.001 | **per-image blend γ0.18 (peak)** ← CURRENT BEST |

γ-sweep (per-image blend): 0.10→0.3878, 0.14→0.3881, **0.18→0.3888**, 0.30→0.3867, 0.42→0.3843. Unimodal, peak γ≈0.18.

---

## WORKS (board-confirmed gains — the winning line)

| approach | effect | notes |
|---|---|---|
| Metric-level denoise (Gaussian blur σ1.0, NLM h0.05) | +0.031 total | the core breakthrough; SSIM is low-pass biased |
| Global mean-face blend β0.20–0.25 | +~0.011 | universal-match hedge |
| ViT 9/11 mean-blend β0.45–0.55 | +~0.002 | brightens (0.31→0.36) + hedges no-signal slots |
| Per-image blend gated by **full** quality signal (rng_frac+0.3·contrast−2·tv), γ0.18 | +0.001–0.0026 | pull low-quality slots harder to mean, keep clean slots; **TV-only signal does NOT work** (pib3 flat) |
| model2 pick-3 slot selection | +0.0088 | |

---

## TAPPED (worked a little then plateaued / redundant — don't grind further)

| approach | result | why stop |
|---|---|---|
| Per-model signal-aware blend (sig1–4) | 0.381–0.3843, all < lowsig | uniform-per-model β is maxed; aggressive hurts |
| Per-image **TV-only** blend (pib3) | 0.385, flat | wrong signal; needs full quality signal |
| ViT mean-blend at high β (0.65/0.85) | 0.3825/0.3796 | over-blend de-diversifies |
| Generative prior — StyleGAN **latent** opt (r1/r2) | mixed (m2/m9 ok, m11 fails); never adopted | not net-positive on board |
| Per-model MEAN blend (vs global mean) | worse | global mean face is the better prior |
| Cluster-blend (blend target = nearest-of-K prototype, K=16) | board −0.003 | global mean wins as target because it's the SMOOTHEST hedge; rougher prototypes help less. Lesson: diversity→base, smoothness→blend target |
| **CNN photometric fix** (m10 brighten 0.278→0.43; m12 desaturate neon) | board: m10phot −0.0014, m12phot −0.0006 (flat) | pib2 photometry already metric-tuned; the dark/neon casts are near-optimal, not errors |

---

## DEAD (board-refuted or structurally impossible — do NOT re-propose)

| approach | result / reason |
|---|---|
| **Faithful/sharp detail recon (`ff`)** | **−0.0703**. Sharp loses to smooth. |
| **Pooled-CNN inverter (upsample-init + TV=8e-3, models 2/3/7)** | **poolab37 −0.0101, poolab237 −0.0144.** Clean-feature +0.12 round-trip did NOT transfer to noisy B=128 features; heavy pooling → smooth blobs → **diversity collapse** under one-to-one. |
| **Extreme-point / archetypal (SPA) MLP selection** (raw + √k-denoised) | **board: archmlp −0.0676, archden −0.0484** (two independent A/Bs). Purity +0.07–0.15 but oracle SSIM Δ→~0 on realistic DP noise; vertices carry per-sample noise → grainy (TV≈0.62). SAME clean→noisy transfer failure as pooling. Purity ≠ SSIM. Collision-peeling NOT built (would inherit same failure). |
| **Batch source separation at B=128** (FastICA, Cocktail-Party CPA, SPEAR++) | no rank knee at 128; refuted by 4 independent lines |
| **ViT gradient inversion / APRIL (9/11)** | APRIL is B=1-only; first-layer grad eff-rank 16/31; mush past batch ~8–64 |
| **Verbatim retrieval from public data** | 0/12 corroborated members; CelebA-200 present but augmented; rowspace ≠ verbatim; right-id/wrong-sample scored neutral (−0.0007) |
| **Trap-bin differencing** | Fowl-*imprint* technique (colinear bins); Boenisch traps have **random** weights → differencing mis-fires and *injects* garbage. Wrong tool. |
| **GIRG (BigGAN, generative weights)** | paper's large-batch win is **BatchNorm-powered**; our models have **no BN** → precondition absent |
| **DLG / iDLG / Geiping pixel or latent optimization at B=128** | collapses to high-cosine NOISE solution (cos 0.98, pure noise); params grow with B |
| Contrast reduction toward gray | worse |
| Weight-decay recovery | ruled out |
| Aggressive/bundled submission | −0.0157 (also: never bundle) |

---

## UNTESTED / OPEN levers (candidates with headroom)

| approach | target | status | expected |
|---|---|---|---|
| ~~Extreme-point / archetypal selection~~ | leaky MLP | **DONE → DEAD** (see above, board −0.048/−0.068) | moved to DEAD |
| ViT diverse **in-distribution** gallery (no mean-blend) on pib2 base | 9/11 | proposed, **NOT built** (deprioritized) | tests diversity-vs-shrink on no-signal slots; ~+0.008 ceiling (2/12 lever) |
| Pin SSIM convention to the grader (win_size/gaussian/σ/channel) | oracle calibration | open | would make offline testing trustworthy |
| Closed-form conv inversion + padding (p=0/p=1) disambiguation | CNN 10/12 | open | fixes possible 1px misalignment |

---

## How to update
- Add a row the moment a submission returns, with the **board** score + `improved` flag.
- Move an approach to DEAD/TAPPED only on a board result, never a local proxy.
- Keep architecture map and current-best line at the top in sync.
