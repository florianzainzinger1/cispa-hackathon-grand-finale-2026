# Integration spec — 3 validated levers for the 0.3926 pipeline

From the other track (analytic + SVD analysis). Each lever is **upstream of your
pib/blur/NLM postproc** (they improve candidate generation), so they compose with
whatever the final stage does. Apply **one at a time, A/B on the board** (never bundle).

Context: our clean reimplementation of your documented pipeline plateaus at **0.354**;
the gap to your 0.3926 is your tuned banks + the pib stage (not in the repo). So these
belong in *your* live pipeline, not ours. Lever 1 is the high-confidence one.

---

## Lever 1 — MLP SVD: use the NOISE-FLOOR rank, not k=128  ★ highest EV

**Why.** The MLP first-layer weight gradient `gW` is `C·X` (rank ≤128) **plus a full-rank
additive-noise defense**. Your `attack_mlp_svd_project.py` picks `k = min(128, eff_rank@1%)`;
but the noise floor sits at ~2.5–5.7% of σ₀ for these models, so `eff_rank@1% ≈ 1024` →
you land on **k=128, keeping ~90 noise directions** (matches your own note: "only 47–78%
of energy in the top-128"). The real face signal is in the **top ~15–40** directions.
Truncating there strips the noise floor. Board-validated **+0.007** on our track; your own
V2 report already names **model8 the biggest denoise win** — and model8's signal is only
~14 dirs, so k=128 vs k=14 is a large difference there.

**Change.** In `attack_mlp_svd_project.py`, replace the `eff_rank@1%` / `k=128` selection
with a noise-floor-relative rank. Drop-in helper:

```python
def noise_floor_k(gW, mult=2.0, cap=128):
    """# singular dirs above mult x the noise-floor (tail median) = recoverable rank."""
    s = torch.linalg.svdvals(gW.float())
    floor = s[len(s)//2:].median()          # tail plateau = pure noise
    return int(min(cap, max(4, (s > mult*floor).sum().item())))
```

Then in `svd_subspace_project(...)` use `k_actual = noise_floor_k(gW)` instead of
`min(k, eff_rank)`. Per-model this yields ≈ **{model1:40, model4:39, model5:37, model8:14}**
(2× floor). Board-validated concrete values were **{1:35, 4:35, 5:37, 8:14}** — model8
MUST be low (~14); do NOT raise 1/4/5 above ~40 (near-noise dirs re-grain).

**Also** in `rec_reconstruct.py` the `REC_DENOISE=1` path hardcodes a rank-**128**
truncation (`topk_right_subspace(gW, 128)`); change that 128 to `noise_floor_k(gW)` too,
for consistency if that path feeds any MLP output.

**Ablation to confirm:** set `mult` large (→k=128) reproduces your current tie; `mult=2`
is the win. A/B one submission: MLP-only swap, all else byte-identical.

---

## Lever 2 — model2 tanh feature scoring  (check first; may be a no-op for you)

**Why.** model2 is a **tanh** CNN whose analytic fc1 readout is clean rank-128, but a
relu-style featureness (penalising negative feature values) flags every legitimately
**negative tanh feature as garbage** → model2 looks dead. We scored against the tanh
output range [-1,1] and it went from clean=0 → 757, a board-validated **+0.0062**.

**Check.** Your `attack_analytic.score_rows` ranks the RGB *view* by `rng_frac × std`
(min-max normalised), and `attack_conv_invert` clamps targets to `[-1,1]` for tanh — so
you may **already** be fine, and your `model2 pick-3 channels` (+0.0088) may be the same
win by another route. Only apply the fix if you score/select model2 feature rows with a
`clamp(-f)` / non-negativity penalty anywhere. The fix, where such scoring exists:

```python
if act == "tanh":   bad = (-f - 1).clamp(min=0) + (f - 1).clamp(min=0)   # range [-1,1]
elif act == "sigmoid": bad = (-f).clamp(min=0) + (f - 1).clamp(min=0)
else:               bad = (-f).clamp(min=0)                              # relu
```

---

## Lever 3 — CNN conv-nullspace chroma  (your flagged "last untested recon lever")

**Why.** The 3→8 conv leaves output colour under-determined; inverting back to 3 channels
fills it with arbitrary chroma. **model10** keeps neon-chroma noise; **model12** has a
low-frequency **sigmoid colour-cast** (you flagged this one yourself — a low-freq error
SSIM rewards correcting). Both are free wins after inversion.

**Change.** In your post-inversion / postproc, per model:

```python
def desaturate(imgs, keep):                 # model10: keep=0.0 (full gray)
    lum = (0.299*imgs[:,0] + 0.587*imgs[:,1] + 0.114*imgs[:,2]).unsqueeze(1)
    return (lum + keep*(imgs - lum)).clamp(0,1)

def colorcast_fix(imgs):                     # model12: neutralise per-channel mean bias
    m = imgs.mean(dim=(0,2,3), keepdim=True)
    return (imgs - m + imgs.mean()).clamp(0,1)
```

Apply `desaturate(x, 0.0)` to model10, `colorcast_fix(x)` to model12. A/B each separately;
model10 desaturation is the bigger visual fix (model12 proves low-sat CNNs score clean).

---

## Order of attack
1. Lever 1 (MLP noise-floor k) — highest EV, targets your flagged MLP headroom.
2. Lever 3 model10 desaturate — big, cheap, isolated.
3. Lever 3 model12 color-cast, then Lever 2 if applicable.

Reference: our `task3/APPROACH.md` (SVD noise-floor analysis) and `task3/merged_pipeline/`
(runnable reference implementation of all four, scores 0.354 on a clean-room base).
