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
export FL_ITERS="${FL_ITERS:-2500}"
export FL_RESTARTS="${FL_RESTARTS:-1}"

echo "GPU job on $(hostname); iters=$FL_ITERS restarts=$FL_RESTARTS models=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u attack_opt.py "$@"
echo "done"
