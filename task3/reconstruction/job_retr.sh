#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
#SBATCH --time=00:25:00
#SBATCH --job-name=retr
# agent2 retrieval. Dedicated GPU node. Usage: sbatch job_retr.sh <stage> [args]
set -e
cd /p/scratch/training2625/zainzinger1/1337/recon_work/code
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=/p/scratch/training2625/zainzinger1/1337/recon_work
export FL_DATA_ROOT=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
export FL_DEVICE=cuda
export PYTHONUNBUFFERED=1
STAGE="${1:-cnncert}"; shift || true
echo "=== stage=$STAGE args=$* $(date) ==="
if [ "$STAGE" = "cnncert" ]; then python -u retr_cnncert.py "$@"
elif [ "$STAGE" = "membership" ]; then python -u retr_membership.py "$@"
elif [ "$STAGE" = "retrieve" ]; then python -u retr_retrieve.py "$@"
else echo "unknown stage $STAGE"; exit 1; fi
echo "=== done $(date) ==="
