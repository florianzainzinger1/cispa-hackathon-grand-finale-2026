#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
set -e
ROOT=/p/scratch/training2625/zainzinger1/1337
cd $ROOT/recon_work/code
source $ROOT/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=$ROOT/recon_work
export FL_DATA_ROOT=$ROOT/FL_Data_Reconstruction
export FL_DEVICE=${FL_DEVICE:-cpu}
export PYTHONUNBUFFERED=1
echo "=== host $(hostname) ==="
echo "=== ${MERGE_CMD:-diag} ==="
eval "${MERGE_CMD:-python -u proxy_labels.py; echo '--- proxy baseline ---'; python -u proxy_score.py; echo '--- postproc TV ---'; python -u merge_postproc.py}"
echo "=== DONE ==="
