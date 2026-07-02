# Local → GitHub → JURECA Workflow

How code gets from this machine onto a JURECA GPU node.

```
┌──────────────┐   git push   ┌──────────┐   git pull   ┌────────────────┐
│ Local repo   │ ───────────► │  GitHub  │ ───────────► │ JURECA scratch │
│ (Claude Code)│              └──────────┘              │  + sbatch job  │
└──────────────┘                                        └────────────────┘
```

## 1. Code locally

Edit code in this repo with Claude Code (or any editor):

```
C:\Users\flori\OneDrive\Dokumente\hackathon\cispaFinale\CISPA-Hackathon
```

Rules:
- Never commit `.venv`, `data/`, `output/`, checkpoints, caches, or secrets
  (the `.gitignore` enforces this).
- Entry point is `main.py` at the repo root — currently a GPU smoke test.
  Replace its contents with the real FL_Data_Reconstruction code.

## 2. Deploy and submit — one command

From **WSL** (or Git Bash):

```bash
cd /mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon
./deploy_submit.sh
```

This script:
1. Commits and pushes local changes (continues safely if nothing changed).
2. SSHes into JURECA as `zainzinger1` using `~/.ssh/id_ed25519_jureca`.
   **If SSH asks for a passphrase or TOTP, type it in the terminal.**
3. Runs `jutil env activate -p training2625`.
4. Clones the repo to `/p/scratch/training2625/zainzinger1/1337/CISPA-Hackathon`
   on first run, otherwise `git pull`s the latest code.
5. Creates `output/` and submits the GPU job: `sbatch job_fl.sh`.
6. Shows the queue: `squeue -u zainzinger1`.

## 3. The Slurm job (`job_fl.sh`)

- Account `training2625`, partition `dc-gpu`, reservation `cispahack`,
  1 node / 1 GPU / 8 CPUs.
- `cd`s into the repo on scratch, activates the task venv:
  `/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv`
- Runs `python main.py`.

Never run training directly on the login node — always go through `sbatch`.

## 4. Checking jobs and logs

On JURECA (`ssh -i ~/.ssh/id_ed25519_jureca zainzinger1@jureca.fz-juelich.de`):

```bash
# Job queue (R = running, PD = pending)
squeue -u zainzinger1

# Logs land in the repo's output/ folder, named after the job id
cd /p/scratch/training2625/zainzinger1/1337/CISPA-Hackathon
ls output
cat output/<jobid>.out   # stdout
cat output/<jobid>.err   # stderr

# Cancel a job if needed
scancel <jobid>
```

Logs stay on JURECA — `output/` is gitignored and never committed.

## 5. Troubleshooting

- **`Permission denied (publickey)`** — check the key is registered in JuDoor
  and you're using `-i ~/.ssh/id_ed25519_jureca`.
- **Clone fails on JURECA** — the GitHub repo must be reachable from JURECA.
  If it's private, either make it public for the hackathon or add a deploy
  key on JURECA (Settings → Deploy keys on GitHub).
- **`jutil: command not found`** — the script uses a login shell (`bash -l`),
  which loads jutil; if it still fails, run `source ~/.bashrc` on JURECA.
- **Job pending forever** — check the reservation:
  `squeue -u zainzinger1 --start` and `scontrol show reservation cispahack`.
