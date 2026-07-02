#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

set -e

cd /p/scratch/training2625/zainzinger1/1337/CISPA-Hackathon
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate

mkdir -p output

python main.py
