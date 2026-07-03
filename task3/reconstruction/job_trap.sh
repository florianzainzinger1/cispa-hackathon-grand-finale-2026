#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
#SBATCH --time=00:20:00
#SBATCH --job-name=trap_L2
# Agent T (lever L2). Runs on a dedicated GPU node (login node is contended).
# Usage: sbatch job_trap.sh <stage> [args...]   stage in {diag,recover,verify}
set -e
cd /p/scratch/training2625/zainzinger1/1337/recon_work/code
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=/p/scratch/training2625/zainzinger1/1337/recon_work
export FL_DATA_ROOT=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
export FL_DEVICE=cuda
export PYTHONUNBUFFERED=1

STAGE="${1:-diag}"
shift || true
echo "=== stage=$STAGE args=$* device=cuda $(date) ==="
if [ "$STAGE" = "diag" ]; then
    python -u trap_diag.py "$@"
elif [ "$STAGE" = "recover" ]; then
    python -u trap_recover.py "$@"
elif [ "$STAGE" = "verify" ]; then
    python -u trap_verify.py "$@"
elif [ "$STAGE" = "all" ]; then
    python -u trap_diag.py
    python -u trap_recover.py
else
    echo "unknown stage $STAGE"; exit 1
fi
echo "=== done $(date) ==="
