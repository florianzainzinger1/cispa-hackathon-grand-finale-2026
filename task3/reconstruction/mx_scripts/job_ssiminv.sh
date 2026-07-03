#!/bin/bash
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
set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
DATA=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
source $DATA/.venv/bin/activate
export FL_WORK_ROOT=$WORK FL_DATA_ROOT=$DATA FL_DEVICE=cuda
cd $WORK/code
python -u mx_ssiminv.py
echo "=== validate ==="
for c in cand_m3_ssiminv cand_m7_ssiminv cand_m2_ssiminv; do
  echo "--- $c ---"
  python -u validate_submission.py $WORK/outputs_mx/$c.pt 2>&1 | tail -1
done
echo "JOB DONE"
