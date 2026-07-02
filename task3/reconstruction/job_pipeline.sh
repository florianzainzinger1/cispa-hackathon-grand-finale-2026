#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err

set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
cd "$WORK/code"
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
mkdir -p "$WORK/output"
export FL_WORK_ROOT="$WORK"
export FL_DEVICE=cuda

echo "== conv-invert (GPU) =="; python -u attack_conv_invert.py
echo "== postprocess ==";       python -u postprocess.py
echo "== face bank ==";         python -u face_bank.py
echo "== build ==";             python -u build_submission.py
echo "== validate ==";          python -u validate_submission.py
echo "pipeline done"
