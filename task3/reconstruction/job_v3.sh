#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --time=00:40:00
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
set -e
cd /p/scratch/training2625/zainzinger1/1337/recon_work/code
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=/p/scratch/training2625/zainzinger1/1337/recon_work
export FL_DEVICE=cuda
MODE=${FL_MODE:-diagnose}
echo "=== v3 job: mode=$MODE models=$* iters=${FL_ITERS} lr=${FL_LR} tv=${FL_TV} ==="
python -u v3_gradmatch.py "$MODE" "$@"
