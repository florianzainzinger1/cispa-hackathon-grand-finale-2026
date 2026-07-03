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
export FL_DEVICE=cpu
export PYTHONUNBUFFERED=1
W=$ROOT/recon_work
echo "=== host $(hostname) ==="
echo "=== SAFE build (-> current_best.pt) ==="
MERGE_MODE=safe python -u merge_build.py
echo "=== AGGRESSIVE build (-> submission_aggressive.pt) ==="
MERGE_MODE=aggr python -u merge_build.py
echo "=== 3-way proxy compare: c3 | safe | aggressive ==="
python -u proxy_score.py $W/outputs_v3/submission_v3.pt $W/outputs_final/submission_current_best.pt $W/outputs_final/submission_aggressive.pt
echo "=== DONE ==="
