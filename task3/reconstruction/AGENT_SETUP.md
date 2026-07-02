# Agent Onboarding — Local ↔ JURECA FL-Reconstruction Setup

This tells another coding agent exactly how to operate in this environment: the
machine topology, how to reach the HPC cluster, the working dev loop, how to run
GPU jobs, where everything lives, and the traps that already cost time. Read it
top to bottom once before touching anything.

---

## 1. Topology (three shells, one cluster)

You are a coding agent on a **Windows 11** laptop. There are three shells and one
remote HPC system:

- **Bash tool = Git Bash** (MSYS). This is what the `Bash` tool runs. It is NOT
  WSL. It mangles Unix-looking paths passed as arguments (see Traps).
- **PowerShell tool** = Windows PowerShell 5.1. Rarely needed here.
- **WSL (Ubuntu)** = the real Linux userland. **The JURECA SSH key, `~/.ssh/config`,
  and the persistent SSH connection all live in WSL.** You reach it by running
  `wsl.exe bash -c '...'` from the Bash tool.
- **JURECA** = FZ Jülich HPC cluster (`jureca.fz-juelich.de`), Slurm-scheduled,
  A100 GPUs. All heavy compute and all the challenge data live here.

Golden rule: **anything involving JURECA goes through WSL**, i.e.
`wsl.exe bash -c 'ssh -o BatchMode=yes jureca "…"'`. Do not try to `ssh` from Git
Bash directly — the key and multiplexing are configured only in WSL.

---

## 2. Reaching JURECA (SSH + TOTP + multiplexing)

Login requires an **SSH key + a TOTP one-time code** (JSC 2-factor). An agent
cannot type TOTP, so the human must open a WSL terminal once per ~8h session and
run `ssh jureca`, enter the code, and leave it connected. That primes a shared
socket (ControlMaster) that all later scripted calls reuse with **no prompt**.

WSL `~/.ssh/config` (already set up):

```
Host jureca jureca.fz-juelich.de
    HostName jureca.fz-juelich.de
    User zainzinger1
    IdentityFile ~/.ssh/id_ed25519_jureca
    ControlMaster auto
    ControlPath ~/.ssh/sockets/cm-%r@%h-%p
    ControlPersist 8h
    ServerAliveInterval 60
```

Test that the shared connection is live (should print a hostname, no prompt):

```bash
wsl.exe bash -c 'ssh -o BatchMode=yes jureca hostname'
```

If it errors with `Permission denied (keyboard-interactive)`, the socket expired
— **stop and ask the human** to run `ssh jureca` + TOTP in their WSL terminal,
then retry. Never try to satisfy TOTP yourself.

GitHub from JURECA: JURECA blocks outbound port 22, so git uses SSH over
`ssh.github.com:443` with a read-only deploy key (configured in JURECA
`~/.ssh/config`). Clone URL must be `git@github.com:pwnXtoXe4/CISPA-Hackathon.git`.

---

## 3. Paths cheat-sheet

| what | path |
|---|---|
| Local repo (Windows) | `C:\Users\flori\OneDrive\Dokumente\hackathon\cispaFinale\CISPA-Hackathon` |
| Local repo (WSL) | `/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon` |
| Pipeline code | `task3/reconstruction/` (in the repo) |
| JURECA team root | `/p/scratch/training2625/zainzinger1/1337` |
| **Challenge data (read-only!)** | `…/1337/FL_Data_Reconstruction/{models,gradients}/model{1..12}.pt` |
| Python venv | `…/1337/FL_Data_Reconstruction/.venv` (`source …/bin/activate`) |
| **My workspace (all outputs)** | `…/1337/recon_work/` → `code/ reports/ candidates/ outputs/ output/` |
| Synced code on JURECA | `…/1337/recon_work/code/` (scp target; runs from here) |
| Slurm logs | `…/1337/recon_work/output/<jobid>.{out,err}` |

Slurm account `training2625`, reservation `cispahack`, partition `dc-gpu`, 1×A100-40GB.

**venv contents:** Python 3.12, `torch 2.12.1+cu130`, numpy, scikit-image (SSIM),
scikit-learn (KMeans), scipy, PIL. **NO** torchvision, matplotlib, or lpips —
render grids with PIL, compute SSIM with skimage.

---

## 4. The dev loop (edit local → run on JURECA → view)

Edit code **locally only** (in the repo). Then sync to JURECA and run. Two helper
scripts live in the session scratchpad; keep this pattern.

**Sync + run (light stages, login-node-safe):** `sync_exec.sh` scps all
`task3/reconstruction/*.py` + `*.sh` to `recon_work/code/`, strips CRLF from `.sh`,
then runs a command in the venv with `FL_WORK_ROOT` set:

```bash
#!/bin/bash
set -e
LOCAL_CODE="/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon/task3/reconstruction"
REMOTE_WORK="/p/scratch/training2625/zainzinger1/1337/recon_work"
VENV="/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate"
CMD="$1"
ssh -o BatchMode=yes jureca "mkdir -p $REMOTE_WORK/{code,reports,candidates,outputs,output}"
scp -q -o BatchMode=yes "$LOCAL_CODE"/*.py "$LOCAL_CODE"/*.sh jureca:"$REMOTE_WORK/code/"
ssh -o BatchMode=yes jureca "cd $REMOTE_WORK/code && sed -i 's/\r\$//' *.sh 2>/dev/null; source $VENV && export FL_WORK_ROOT=$REMOTE_WORK && $CMD"
```

Invoke it (note the wrapping — see Traps):

```bash
wsl.exe bash -c 'bash "/mnt/c/…/scratchpad/sync_exec.sh" "python -u inspect_models.py"'
```

To sync just ONE new file without touching others:
`wsl.exe bash -c 'scp -q -o BatchMode=yes "/mnt/c/…/task3/reconstruction/foo.py" jureca:…/recon_work/code/foo.py && ssh -o BatchMode=yes jureca "cd …/recon_work/code && source $VENV && python -u foo.py"'`

**Viewing images:** you CAN see PNGs. Pull them to the local scratchpad with `scp`,
then use the `Read` tool on the `.png` — it renders inline. Generate grids on
JURECA with `utils.save_image_grid` (PIL). This is how you judge reconstruction
quality; do it liberally.

Code is env-driven: `FL_DATA_ROOT`, `FL_WORK_ROOT`, and `FL_DEVICE` (`cpu`/`cuda`)
override paths/device. Default device is CPU; sbatch sets `FL_DEVICE=cuda`.

---

## 5. Running GPU jobs (sbatch)

Heavy work (any optimization / gradient matching / large inversions) MUST go
through Slurm, never the login node. Job scripts live in `task3/reconstruction/`
(`job_pipeline.sh`, `job_finalize.sh`, `job_recon.sh`). SBATCH header pattern:

```
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
```

Submit + watch (pass env overrides via `--export`):

```bash
wsl.exe bash -c 'bash "/mnt/c/…/sync_exec.sh" "cd …/recon_work/code && sbatch --export=ALL,FL_ITERS=2000 job_recon.sh model9 && squeue -u zainzinger1"'
```

Monitor: `squeue -u zainzinger1`; history/durations: `sacct -X --starttime today
--format=JobID,JobName,State,Elapsed`. Poll a job to completion with a small
loop-in-a-script (not inline). Typical GPU stage here is 1–4 min.

---

## 6. The task (context)

Federated-learning **gradient inversion**: reconstruct the 128 private 64×64 RGB
face images for each of 12 models (1536 total) from one client gradient per model.
Scored by per-model **Hungarian one-to-one SSIM** (leaderboard 30% / final 70%),
so **quality AND diversity** both matter — duplicating one good face scores poorly.

Each `gradients/model{i}.pt` = `{gradients:{param→tensor}, family, activation,
feature_shape, batch_size=128}`. Each `models/model{i}.pt` = plain `state_dict`.
Families: MLP (1,4,5,8), CNN (2,3,6,7,10,12), ViT (9,11). Dataset ≈ CelebA faces.

**Submission:** `.pt` dict, keys exactly `model1`..`model12`, each
`Tensor(128,3,64,64)` float32 in `[0,1]`, finite, no exact duplicates, ≤300 MB.
Endpoint `POST http://<SUBMIT_SERVER>/submit/21-fl-audit` with header `X-API-Key`.
**The human submits** — an agent must NOT enter the API key (hard rule).

Full method + per-model triage is in `task3/reconstruction/README.md`. Current
best = `recon_work/outputs/submission.pt` (v1, leaderboard 0.2787, backup
`submission_v1_bank.pt`). Shared-identity across models was tested and **refuted**
(`reports/shared_identity_test.md`) — do not attempt cross-model face reuse.

---

## 7. Traps that already cost time (read these!)

1. **Inline `$VAR` through `wsl.exe bash -c '…'` gets eaten** → `scp`/`mkdir`
   receive empty strings. ANY command with shell variables or `$(…)` must be in a
   **script file** run as `wsl.exe bash -c 'bash "/mnt/c/…/script.sh"'`. One-liners
   without variables are fine inline.
2. **Nested SSH quoting**: `ssh jureca "… ( … ) …"` breaks on parens/redirects.
   Put the remote body in a local `.sh` and pipe it: `ssh jureca "bash -s" < script.sh`.
3. **Git Bash path mangling**: `wsl.exe bash /mnt/c/…/x.sh` becomes
   `C:/Program Files/Git/mnt/…`. Always wrap: `wsl.exe bash -c 'bash "/mnt/c/…/x.sh"'`.
4. **CRLF on `.sh`**: files written on Windows get `\r`; `sbatch`/shebang break.
   `sed -i 's/\r$//' *.sh` after copying (sync_exec does this).
5. **Login-node compute**: do NOT run optimization/inversion on the login node.
   Early on, three CPU inversion runs stacked up there and had to be killed. Use
   sbatch. Analytic ratio math and small analyses (loading .pt, KMeans, a few
   thousand SSIM calls) are OK on the login node — keep them short and single.
6. **CUDA is visible on the login node** but must not be used there; code defaults
   to `FL_DEVICE=cpu` and sbatch flips it to `cuda`.
7. **Non-determinism**: `attack_conv_invert.invert_conv` inits `torch.rand` with no
   seed → CNN inversions differ between runs. Analytic (MLP) and KMeans
   (`random_state=0`) are deterministic. To reproduce a submission from existing
   candidate banks, just re-run `postprocess → face_bank → build_submission`.
8. **ViT gradient matching reaches cosine ~0.9 but yields incoherent color-patches**
   — high gradient cosine ≠ good image for ViT. Models 9/11 use a face-bank
   fallback; don't burn GPU trying to invert them.

---

## 8. Safety / house rules

- Edit code **locally**; never edit on JURECA directly (it gets overwritten by scp).
- **Never modify or commit** the challenge data (`models/`, `gradients/`) or
  `.venv`, `data/`, `output(s)/`, checkpoints, caches, or secrets. `.gitignore`
  covers `*.pt`, `.env`, etc.
- **Never overwrite** `recon_work/outputs/submission.pt` or `submission_v1_bank.pt`.
  New experiments write to `recon_work/outputs_v2/` and `candidates_v2/`.
- **Never handle the API key / HF / GitHub tokens / passwords / TOTP.** The human
  runs the actual submission. Never print secrets.
- Ask before destructive ops (`rm -rf`, `git reset --hard`). Never `chmod 777`.
- Commit code with plain `git add/commit/push` (does NOT trigger any job). Only
  `deploy_submit.sh` triggers a JURECA pull + sbatch, so avoid it unless intended.

---

## 9. 60-second quick start

```bash
# 0. confirm the multiplexed connection is live (else ask human for TOTP)
wsl.exe bash -c 'ssh -o BatchMode=yes jureca hostname'

# 1. see the data / current submission
wsl.exe bash -c 'ssh -o BatchMode=yes jureca "ls /p/scratch/training2625/zainzinger1/1337/recon_work/outputs; ls /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/gradients"'

# 2. edit code locally in task3/reconstruction/, then run a light stage:
wsl.exe bash -c 'bash "/mnt/c/…/scratchpad/sync_exec.sh" "python -u validate_submission.py /p/scratch/training2625/zainzinger1/1337/recon_work/outputs/submission.pt"'

# 3. heavy stage → sbatch job_*.sh, then squeue/sacct, then scp PNGs back and Read them.
```

Pipeline stages (all in `task3/reconstruction/`): `inspect_models.py` →
`attack_analytic.py` → `attack_conv_invert.py` → `infer_labels.py` →
`postprocess.py` → `face_bank.py` → `build_submission.py` → `validate_submission.py`
(orchestrated by `run_pipeline.sh`; GPU opt via `job_recon.sh`).
