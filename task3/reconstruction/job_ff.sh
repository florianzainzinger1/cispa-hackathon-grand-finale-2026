#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
set -e
export OMP_NUM_THREADS=8
ROOT=/p/scratch/training2625/zainzinger1/1337
export FL_WORK_ROOT=$ROOT/recon_work
export FL_DATA_ROOT=$ROOT/FL_Data_Reconstruction
source $FL_DATA_ROOT/.venv/bin/activate
cd $FL_WORK_ROOT/code
python -u ff_build.py
echo "FF_JOB_DONE"
