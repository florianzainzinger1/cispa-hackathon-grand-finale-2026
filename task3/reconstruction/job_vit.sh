#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
#SBATCH --time=00:25:00
#SBATCH --job-name=vit_april
# agent2 ViT (models 9,11). Dedicated GPU node (login node is contended).
# Usage: sbatch job_vit.sh <stage> [args]   stage in {inspect,april,verify}
set -e
cd /p/scratch/training2625/zainzinger1/1337/recon_work/code
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=/p/scratch/training2625/zainzinger1/1337/recon_work
export FL_DATA_ROOT=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
export FL_DEVICE=cuda
export PYTHONUNBUFFERED=1
STAGE="${1:-april}"; shift || true
echo "=== stage=$STAGE args=$* $(date) ==="
if [ "$STAGE" = "inspect" ]; then python -u vit_inspect.py "$@"
elif [ "$STAGE" = "april" ]; then python -u vit_april.py "$@"
elif [ "$STAGE" = "verify" ]; then python -u vit_verify.py "$@"
else echo "unknown stage $STAGE"; exit 1; fi
echo "=== done $(date) ==="
