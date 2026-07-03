# Task 3 — Coordination Hub (FRESH PUSH, target: beat 0.48)

**This file is the shared brain for the multi-agent push. The main/coordinator agent owns and edits it.**
Worker agents READ it fully before touching anything, then work only inside their namespace.

> **Rewritten 2026-07-03.** This supersedes the earlier 5-agent → 3-agent iterations (old PARTS 0/0.5/0.6).
> Those levers have been RUN and their verdicts are now baked into the ledger below. Do not resurrect a
> lever that the DEAD/TAPPED ledger (§3) has already buried — read it first, it is half the value of this doc.

---

## §0 — LIVE STATE (read this first)

**Standing best = `0.3722` (leaderboard-confirmed).** File on JURECA:
`recon_work/outputs_final/test_nlm05.pt` (= c3 base + model2 pick-3 channels + per-image edge-preserving
NLM denoise h≈0.05). It is VALID (12 models × 128 distinct 64×64 RGB, ranges in [0,1]).

**Canonical pointer is now correct.** `recon_work/outputs_final/submission_current_best.pt` was repointed
to the 0.3722 content on 2026-07-03 (it had been stale at ≈0.3325). The previous safe build is preserved at
`outputs_final/submission_prev_safe.pt`; frozen true-c3 is `outputs_v3/submission_v3.pt` (0.3325 fallback).

**Environment = GO** (verified 2026-07-03): JURECA reachable via WSL ControlMaster (`jrlogin10`); challenge
data present (12 models + 12 gradients); venv intact (`torch 2.12.1+cu130`); reservation `cispahack` live;
93 code files synced in `recon_work/code/`; all lever outputs on disk (`outputs_{trap,retr,bss,gp,vit,final}`).

**Leader ~0.48. Final rank = hidden 70% split** → every method must GENERALIZE, not fit the public 30%.

### Submission is now AGENT-RUN (authorization changed)
The human EXPLICITLY authorized the agent to submit directly. Mechanism (key never printed, never in repo):
```
FL_API_KEY='<key>' bash submit.sh <path-to.pt>      # posts to http://<SUBMIT_SERVER>/submit/21-fl-audit, header X-API-Key
```
Response JSON = `{submission_id, score, previous_best, delta, improved, message}`. **Cooldown = 5 min on a
successful submit** (server returns 429 "Wait N seconds" if early; a `requests` loop that parses the wait and
retries automates it). Leaderboard keeps **best-only** → a regression cannot hurt the standing best, so
single-variable testing has NO downside. **Never echo the key into visible output or logs.**

---

## §1 — THE ONE LAW: only the leaderboard is truth

This is the hardest-won lesson of the whole project. **The local proxies (subspace residual, gradient
cosine, capture %, diversity/TV) DO NOT track the SSIM score and repeatedly misled us into regressions.**
Concretely:
- `v3` hit gradient-cosine 0.98 on **pure noise**. Cosine is necessary-not-sufficient — use it only to REJECT.
- The subspace-residual-certified "faithful" recon (`ff`) scored **0.3018 (−0.0703)** — worse than blur.
- Residual-certified model10 swaps came out flat/negative on the board.

**Operating rules for every agent (violating these is how we lost hours):**
1. **Test ONE variable per submission.** Take the current best, change exactly one thing, submit. `delta × 12`
   ≈ that model's true per-model effect. **NEVER bundle** — every bundle we ever tried regressed.
2. **Keep only leaderboard-confirmed wins.** A proxy improvement is a hypothesis, not a result.
3. **Faithful detail LOSES to smooth blur on this metric** (confirmed 3×: c4 net-zero, c5 −0.0045, ff −0.070).
   SSIM is low-pass biased; grain is the killer. Do not chase pixel-faithfulness for its own sake.
4. **Diversity is NOT free** under Hungarian one-to-one: filler faces score ~0.15–0.35, so "128 uniform
   ~0.42" beats "60@0.60 + 68@0.20 ~0.39" unless the excellent tier is >0.7 AND the fillers would be noise.

---

## §2 — Confirmed scoreboard (leaderboard, public 30% split)

| approach | score | delta | what it was | verdict |
|---|---:|---:|---|---|
| c1 | 0.2787 | — | analytic ratio + robust-mean, noise-pad | superseded |
| c3 | 0.3325 | +0.0538 | coverage-first analytic + never-noise diverse bank | base of everything (`outputs_v3/submission_v3.pt`) |
| c4 | ~0.332 | −0.0003 | c3 + strided CNN 2/3/7 re-inverted higher base-res | **strided CNN tapped out** |
| c5 | 0.328 | −0.0045 | c3 + model6 rebuilt (56 real recon + 72 bank) | **grain loses to clean bank** |
| ff | 0.3018 | −0.0703 | 9 models swapped to faithful low-residual recon + per-family denoise | **faithful detail loses to blur** |
| **nlm05** | **0.3722** | **+0.0397** | c3 + model2 pick-3 + edge-preserving NLM h≈0.05 | **CURRENT BEST** (`outputs_final/test_nlm05.pt`) |
| Leader | ~0.48 | — | other team | target |

**Anatomy of the +0.0397 (nlm05 over c3):** almost all of it was **metric-level**, not reconstruction —
global blur σ=1.0 alone was +0.0259; edge-preserving NLM (keeps face edges, kills grain) beat blur by +0.0045;
the one real reconstruction win was model2 pick-3-channels (+0.0088). NLM h≥0.10 over-smooths; bilateral ≈ NLM;
per-model blur was slightly worse than uniform. **Classical denoise/blur is now TAPPED.**

---

## §3 — DEAD / TAPPED ledger (do NOT re-run these)

Every one of these was actually executed and measured this project. Re-running them is pure waste.

| lever | status | one-line reason | evidence |
|---|---|---|---|
| **Retrieval (verbatim public data)** | **DEAD** | 0/12 corroborated verbatim hits; source is CelebA top-200 (Fantômas family) but **augmented/re-rendered** — no clean original passes a certificate | `reports/retr/STATUS.md`, `membership.json` (max SSIM 0.51 @ cc178, no >0.95 spike), [[fl-recon-retrieval]] |
| **BSS at B=128 (SPEAR++ / CPA)** | **REFUTED** | SPEAR++ 0%-accuracy cliff at exactly m=1024/b=128; CPA breaks below our noise; dies on GELU model8 | RESEARCH_FINDINGS §3, only `outputs_bss/bss_model5.pt` ever produced |
| **ViT 9/11 gradient inversion (APRIL)** | **DEAD** | APRIL is a B=1 attack; averaged patch-embed grad has eff-rank ~16/~31 → recovers a batch-MEAN face, not identity; B=128 best-per-sample SSIM 0.19 < bank | `reports/vit/STATUS.md`, [[fl-recon-vit-april]] |
| **StyleGAN latent-matching / GIFD synthesis** | **DEAD** | realistic but wrong-identity faces; −0.0093 on model2/9; GIFD's 0.577 is a B≤4 number | [[fl-recon-r1-generative-prior]], RESEARCH_FINDINGS §3 |
| **Faithful analytic recon (low-residual detail)** | **DEAD** | ff = −0.0703; detail loses to blur under SSIM | scoreboard §2 |
| **"Recover more noisy faces" (c4/c5)** | **DEAD** | SSIM punishes grain > wrong-identity bank; net-zero/negative | scoreboard §2 |
| **Batch-128 pixel gradient-matching** | **DEAD** | pure noise, all families | prior sessions |
| **Per-model blur / mean-std to `gallery/celeba_top200_*.pt`** | **TAPPED/INVALID** | per-model blur < uniform; that gallery is 99% white (mean 0.99) — the +0.2 it showed was a broken-gallery artifact | [[fl-recon-metric-breakthrough]] |
| **Classical denoise (NLM/blur/bilateral/TV)** | **TAPPED** | fully swept; NLM h=0.05 is the optimum = current best | scoreboard §2 |

**The one recon lever that is theoretically-open-but-empirically-doubtful:** true batch source-separation of
the MLP blends (models 1/4/5) to split neurons that mix multiple faces. It is the "capture ≠ SSIM" gap. BUT it
is refuted at B=128 (above) AND the ff result says even a perfect separation would need heavy denoise to score,
capping its payoff near where blur already sits. **LOW priority; only as a blurred-then-submitted source, only
after §4 is exhausted.**

---

## §4 — The genuinely-live levers (honest EV, all UNTESTED on the board)

We are at a hard spot: the easy levers are gone and 0.3722→0.48 is a big gap. These three are the honest
remaining positive-EV moves. Each is cheap and settles in 1–2 single-variable leaderboard calls.

**L-A — Distribution-matched BLURRED bank for pure-fill slots. (TOP EV.)**
~256 of the 1536 slots (ViT 9 + ViT 11) are currently **generic** diverse-bank faces scoring ~0.15–0.25; add
the model6/8/5 bank tails and it's more. The data family is confirmed CelebA **top-200 identities** and the
empirically-closest crop is **cc178→64** (membership.json: CelebA cc178 faces hit SSIM 0.51 vs 0.27 within-set
baseline — genuinely closer to the private data than random). Build a diverse 128-face bank from CelebA top-200
at cc178→64, **blur it to the SSIM optimum** (σ≈1.0, the winning global-blur setting), and swap it in for ViT
9/11. Under Hungarian one-to-one SSIM this is bank-vs-bank: an *in-distribution, blurred* bank should out-score
a *generic, blurred* bank because SSIM rewards low-freq structural match to the true faces. **This is the single
most defensible untested lever** and it targets exactly the score-limiting models the build agent flagged
(model6/8, ViT 9/11 "unchanged — real gains require these"). Caveat that does NOT apply here: the "wrong-identity
realistic faces hurt" lesson was measured against a REAL reconstruction (StyleGAN on models with signal); ViT
has zero signal, so there is no real recon to lose to — only a better vs worse bank.

**L-F — Correct global photometric matching. (Cheap, medium-low EV.)**
The earlier mean/std-match test was INVALID (it matched to the 99%-white broken gallery). A *correct* CelebA-64
reference (proper faces, mean ≈0.40) has never been tested. On the current best, try per-channel mean/std match
+ gamma/contrast match to that reference — SSIM's luminance and contrast terms reward matching the true
brightness/contrast distribution. One submission to settle it.

**L-B — Global geometric probe. (Cheap, low-medium EV.)**
SSIM is shift-sensitive. If our recons are systematically off-center vs the GT crop, a small global recenter
helps; if already centered, it hurts. Blind, but 1–2 leaderboard probes (global shift ±1–2px, mild recenter/
rescale) settle it cheaply. Do NOT bundle with anything.

**Also cheap-confirm (likely tapped):** stack a light blur *on top of* nlm05 (base changed since the sweep);
re-tune blur σ on the nlm05 base once. One test; expect ~0 but it's free.

**Sober note.** If L-A/L-F/L-B all come back flat, the honest read is that 0.48 required something structural we
do not have (the leader likely had the real dataset, or a fundamentally stronger attack). At that point the
right move is to lock in 0.3722 and stop burning cooldowns — say so plainly rather than inventing busywork.

---

## §5 — Fresh 3-agent plan (paste-ready briefs)

Three focused agents, each with an isolated namespace (own file-prefix + output dir → no collisions). Agent C
owns the canonical best and is the ONLY one that submits (serializes the 5-min cooldown; single-variable
discipline). A and B produce *candidate .pt files* and hand them to C with a one-line hypothesis; C submits,
logs the delta, keeps only confirmed wins.

| agent | role | code prefix | outputs dir | report |
|---|---|---|---|---|
| **A** | distribution-matched bank + fill quality (L-A) | `bank_*.py` | `recon_work/outputs_bank/`, `recon_work/gallery/` | `reports/bank/STATUS.md` |
| **B** | metric + geometry squeeze (L-F, L-B) | `mx_*.py` | `recon_work/outputs_mx/` | `reports/mx/STATUS.md` |
| **C** | integrate + submit + scoreboard (owns best) | `merge_*.py`,`sub_*.py` | `recon_work/outputs_final/` | `reports/build/STATUS.md`, `reports/SCOREBOARD.md` |

Shared rules: read §1 (the one law) before anything. Never touch the challenge `models/`,`gradients/`,`.venv`.
Never overwrite `outputs_v3/submission_v3.pt` (c3 fallback) or `outputs_final/test_nlm05.pt` (the 0.3722 best).
Edit code LOCALLY in `task3/reconstruction/`, sync + run on JURECA. scp ONLY your own prefix into `code/`
(never `scp *.py` — it clobbers other agents). Heavy runs go through `sbatch`; light SVD/denoise/gallery on the
login node is fine. See §6 for ops.

### ▶ CHAT A — Distribution-matched bank (L-A, TOP priority)
```
You are Agent A on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md FULLY, especially §1
(only the leaderboard is truth), §3 (dead ledger — do not re-run those), and §4 L-A. Namespace: code bank_*.py,
outputs recon_work/outputs_bank/ + recon_work/gallery/, report reports/bank/STATUS.md. Do NOT touch other
agents' files or the challenge data/venv.

GOAL: beat the generic diverse bank on the PURE-FILL models (ViT 9 & 11 first; then model6/8 tails) with an
in-distribution, blurred CelebA-top-200 bank.

STEPS:
1. Build a CLEAN CelebA top-200-identity gallery at cc178->64 (CenterCrop(178)->Resize(64)->ToTensor, [0,1]),
   plus cc148->64 as an alternate. Verify it is real faces (mean ~0.40, NOT 99% white — the old
   gallery/celeba_top200_*.pt is broken, mean 0.99; do not reuse it). Login node has internet
   (flwrlabs/celeba). Isolated deps via uv pip install --target recon_work/runs/pylibs if needed.
2. Pick a DIVERSE 128-face subset (farthest-point / k-center on pixel or a cheap embedding) so the bank covers
   the face-space, one face per identity where possible.
3. Blur the bank to the SSIM optimum (start Gaussian sigma=1.0 — the winning global-blur setting; also make a
   sigma=0.8 and a NLM-h0.05 variant). Save as recon_work/outputs_bank/bank_vit_{sigma}.pt etc.
4. Produce a candidate submission = current best (outputs_final/test_nlm05.pt) with ONLY model9 & model11
   swapped to the new bank. Hand this ONE file to Agent C with the hypothesis "in-distribution blurred bank >
   generic bank on ViT". Do NOT submit yourself.
5. On Agent C's confirmed delta: if +, extend the same bank to the model6/model8/model5 bank-fill tails (again
   one model per submission via C). If -, the generic bank is already near-optimal for fill — say so and stop.

Measurement is leaderboard-only (via C). Local proxies do not track SSIM — use them only to sanity-check
validity (128 distinct, [0,1], finite). Report per-swap deltas in reports/bank/STATUS.md.
```

### ▶ CHAT B — Metric + geometry squeeze (L-F, L-B)
```
You are Agent B on CISPA Hackathon Task 3. Read task3/reconstruction/AGENT_SETUP.md FULLY, especially §1, §3,
and §4 L-F/L-B. Namespace: code mx_*.py, outputs recon_work/outputs_mx/, report reports/mx/STATUS.md.

GOAL: squeeze free SSIM out of the current best (outputs_final/test_nlm05.pt, 0.3722) via correct global
photometric + geometric transforms. Every candidate changes exactly ONE variable.

TASKS (each produces one candidate .pt handed to Agent C; NEVER bundle):
1. Build a CORRECT CelebA-64 photometric reference (per-channel mean/std/histogram of real cc178->64 faces,
   mean ~0.40). The old mean/std test was invalid (broken 99%-white gallery) — this is untested.
2. mx_photometric.py: current best with per-channel mean/std match to that reference. One candidate. Then a
   separate gamma/contrast-match candidate. (Separate submissions — do not stack.)
3. mx_geometry.py: global recenter/shift probes on the current best — one candidate per {shift +1px, shift
   -1px, mild 2% center-zoom}. SSIM is shift-sensitive; this tests whether recons are off-center vs the GT crop.
4. Cheap-confirm (likely tapped, but free): one candidate = light Gaussian blur (sigma 0.5) stacked on top of
   nlm05, and one = blur sigma re-tuned on the nlm05 base.

Hand each candidate to Agent C with a one-line hypothesis. Keep only leaderboard-confirmed wins. If everything
is flat, report "metric/geometry tapped" clearly. reports/mx/STATUS.md tracks every candidate + its delta.
```

### ▶ CHAT C — Integrate, submit, scoreboard (owns the best)
```
You are Agent C on CISPA Hackathon Task 3, the integration + submission owner. Read
task3/reconstruction/AGENT_SETUP.md FULLY. You OWN recon_work/outputs_final/submission_current_best.pt (currently
= test_nlm05.pt, 0.3722) and reports/SCOREBOARD.md. Namespace: merge_*.py, sub_*.py, outputs_final/.
You are the ONLY agent that submits — this serializes the 5-min cooldown and enforces single-variable discipline.

SUBMISSION: FL_API_KEY='<key>' bash submit.sh <file>  (POST http://<SUBMIT_SERVER>/submit/21-fl-audit, X-API-Key).
The human/coordinator provides FL_API_KEY. NEVER echo the key. Write sub_submit.py: a requests loop that POSTs,
parses the JSON {score,delta,improved,previous_best}, and on 429 parses "Wait N seconds" and retries after the
cooldown. Log EVERY response to reports/SCOREBOARD.md (append-only submission log).

RULES:
1. Validate every candidate with validate_submission.py (12 models, 128 distinct, [0,1], finite) BEFORE submit.
2. Submit exactly ONE single-variable candidate at a time. Never bundle. After each result:
   - improved=true  -> copy that candidate over submission_current_best.pt (back up the prior as
     submission_prev_safe.pt), update SCOREBOARD, tell A/B the new best.
   - improved=false -> discard the candidate; standing best is untouched (best-only board). Log the delta so we
     never retry it.
3. Prioritize the queue: A's ViT-bank swap first (highest EV), then B's photometric, then B's geometry.
4. Keep outputs_final/test_nlm05.pt and outputs_v3/submission_v3.pt frozen as fallbacks. Never overwrite them.
5. If the whole queue comes back flat, STOP submitting, lock in 0.3722, and report that plainly. Do not burn
   cooldowns on hail-marys.
```

---

## §6 — Ops cheat-sheet (every agent needs this)

**Topology.** Windows 11 laptop; JURECA = FZ-Jülich Slurm cluster (A100). **Everything JURECA goes through WSL**
(`wsl.exe bash -c '...'`) — WSL holds the SSH key/config/ControlMaster socket. The `Bash` tool is Git Bash and
mangles unix paths; always wrap JURECA calls in `wsl.exe bash -c '...'`.

**Reach JURECA.** Login needs SSH key + TOTP; the human runs `ssh jureca` once per ~8h in a WSL terminal to
prime the ControlMaster socket, then scripted calls reuse it. Test:
`wsl.exe bash -c 'ssh -o BatchMode=yes jureca hostname'` → prints `jrlogin10.jureca`. If
`Permission denied (keyboard-interactive)` → socket expired, **ask the human for TOTP** (never satisfy TOTP
yourself). **Login node HAS internet** (HF/pip/git via 443); **compute nodes are offline** (pre-download first).

**Paths.**
| what | path |
|---|---|
| Local repo (Win) | `C:\Users\flori\OneDrive\Dokumente\hackathon\cispaFinale\CISPA-Hackathon` |
| Local repo (WSL) | `/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon` |
| Pipeline code (local, edit here) | `task3/reconstruction/` |
| JURECA team root | `/p/scratch/training2625/zainzinger1/1337` |
| **Challenge data (READ-ONLY)** | `…/1337/FL_Data_Reconstruction/{models,gradients}/model{1..12}.pt` |
| venv | `…/1337/FL_Data_Reconstruction/.venv` |
| **Workspace (all outputs)** | `…/1337/recon_work/` |
| Synced code (scp target; run from here) | `…/1337/recon_work/code/` |
| Slurm logs | `…/1337/recon_work/output/<jobid>.{out,err}` |
| **Current best** | `…/recon_work/outputs_final/test_nlm05.pt` = `submission_current_best.pt` (0.3722) |
| c3 fallback (frozen) | `…/recon_work/outputs_v3/submission_v3.pt` (0.3325) |

Slurm: account `training2625`, reservation `cispahack`, partition `dc-gpu`, 1×A100-40GB. venv: Python 3.12,
`torch 2.12.1+cu130`, numpy, scikit-image (SSIM), scikit-learn (FastICA), scipy, PIL. **NO torchvision / lpips /
matplotlib.** Extra deps → isolated `uv pip install --target recon_work/runs/pylibs <pkg>` then
`sys.path.insert(0,'…/pylibs')`.

**Dev loop.** Edit code LOCALLY → scp your prefix to `recon_work/code/` (strip CRLF) → run in the venv with
`FL_WORK_ROOT`/`FL_DATA_ROOT`/`FL_DEVICE` set. Single-file pattern:
```
wsl.exe bash -c 'scp -q -o BatchMode=yes "/mnt/c/…/bank_foo.py" jureca:…/recon_work/code/ && \
  ssh -o BatchMode=yes jureca "cd …/recon_work/code && source …/.venv/bin/activate && \
  export FL_WORK_ROOT=…/recon_work FL_DATA_ROOT=…/FL_Data_Reconstruction FL_DEVICE=cpu && python -u bank_foo.py"'
```
For anything with shell vars / parens / redirects, put it in a `.sh` file and pipe it:
`ssh jureca "bash -s" < script.sh` (nested inline quoting breaks). **View images:** scp the PNG to local
scratchpad, then `Read` it (renders inline). Grids via `utils.save_image_grid`.

**GPU jobs (sbatch) — REQUIRED for all real work.** Every reconstruction / optimization / inversion / diffusion / ensembling / ICA run goes through `sbatch` on the GPU node, NOT the login node. Only trivial (<~10s) `.pt` assembly, validation, montage rendering, and single SVD/denoise passes may run on the login node. **GPU path VERIFIED 2026-07-03** (job 15402755 on `jrc0227`: 4× A100-40GB, `torch 2.12.1+cu130`, `cuda_available=True`, ~41.6 GB free). Header (each directive on its own line; strip CRLF with `sed -i 's/\r$//'` before sbatch):
```
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
```
In the job body: `source $ROOT/FL_Data_Reconstruction/.venv/bin/activate`, set `FL_DEVICE=cuda`. Monitor `squeue -u zainzinger1`; wait+read via a piped `.sh` (`ssh jureca "bash -s" < wait.sh`) — inline nested SSH quoting breaks on `$(...)`/redirects. History `sacct -X -j <id> --format=JobID,State,Elapsed,ExitCode`.

**Traps that already cost time.** (1) Inline `$VAR`/`$(...)` through `wsl.exe bash -c '…'` gets eaten — use a
script file. (2) Nested SSH quoting breaks on parens/redirects — pipe a local `.sh`. (3) Git Bash mangles paths
— always wrap `wsl.exe bash -c`. (4) CRLF on `.sh` breaks sbatch — `sed -i 's/\r$//' *.sh`. (5) Don't stack CPU
jobs on the login node. (6) `attack_conv_invert.invert_conv` is unseeded → CNN inversions differ per run.

**House rules.** Edit locally, never on JURECA (scp overwrites). Never modify/commit challenge
`models/`,`gradients/`,`.venv`,`data/`. Never overwrite `test_nlm05.pt`, `submission_v3.pt`, or
`submission_v1_bank.pt`. The agent MAY submit (§0) but must never print the API key. Ask before `rm -rf` /
`git reset --hard`. Plain `git add/commit/push` is safe.

---

## §7 — Escalate to the coordinator (this chat) if
- A live lever (§4) is confirmed dead on the board → prune it, re-point agents.
- Two candidates want the same slot / a cross-cutting decision is needed.
- The whole §4 queue comes back flat → decision to lock 0.3722 and stop.
- The ControlMaster socket expired (needs human TOTP) or a submission returns a non-cooldown error.
