#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/runs/r1/logs/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/runs/r1/logs/%j.err
set -e
ROOT=/p/scratch/training2625/zainzinger1/1337
WORK=$ROOT/recon_work
CODE=$WORK/code
source "$ROOT/FL_Data_Reconstruction/.venv/bin/activate"
export FL_DEVICE=cuda
export FL_DATA_ROOT="$ROOT/FL_Data_Reconstruction"
export FL_WORK_ROOT="$WORK"
export FL_SG_REPO="$WORK/runs/repos/stylegan2-ada-pytorch"
export FL_SG_PKL="$WORK/runs/weights/celebahq256.pkl"
export PYTHONPATH="$WORK/runs/pylibs:$FL_SG_REPO:$CODE"
cd "$CODE"
echo "host=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u attack_stylegan.py "$@"
echo "JOB_DONE"
