#!/bin/bash
set -e

LOCAL_REPO="/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon"
REMOTE_USER="zainzinger1"
REMOTE_HOST="jureca.fz-juelich.de"
REMOTE_BASE="/p/scratch/training2625/zainzinger1/1337"
REMOTE_REPO="/p/scratch/training2625/zainzinger1/1337/CISPA-Hackathon"
SSH_KEY="$HOME/.ssh/id_ed25519_jureca"

# Allow running from Git Bash on Windows too (no /mnt/c there)
if [ ! -d "$LOCAL_REPO" ]; then
    LOCAL_REPO="/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon"
fi

cd "$LOCAL_REPO"

echo "=== Local git status ==="
git status

echo "=== Commit and push ==="
git add .
git commit -m "auto update before JURECA submit" || echo "Nothing to commit"
git push

REPO_URL=$(git remote get-url origin)
# JURECA blocks outbound port 22 and has no HTTPS credentials, so the remote
# side uses SSH via ssh.github.com:443 with a read-only deploy key
# (configured in ~/.ssh/config on JURECA).
REMOTE_CLONE_URL="git@github.com:pwnXtoXe4/CISPA-Hackathon.git"

echo "=== SSH to JURECA, pull latest code, submit job ==="
# Variables below are expanded LOCALLY (unquoted heredoc), then the script
# runs on JURECA in a login shell. If SSH prompts for a passphrase/TOTP,
# type it in this terminal.
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" /bin/bash -l << EOF
set -e
jutil env activate -p training2625
cd "$REMOTE_BASE"

if [ ! -d "$REMOTE_REPO/.git" ]; then
    echo "Remote repo not found. Cloning..."
    git clone "$REMOTE_CLONE_URL" "$REMOTE_REPO"
fi

cd "$REMOTE_REPO"
git remote set-url origin "$REMOTE_CLONE_URL"
git pull
mkdir -p output
sbatch job_fl.sh
squeue -u "$REMOTE_USER"
EOF
