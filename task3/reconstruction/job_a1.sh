#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
#SBATCH --time=00:30:00
#SBATCH --job-name=a1_svd_vit

# a1 experiment: SVD subspace projection (MLP) + ViT analytic inversion
# Login-node safe parts (SVD, clustering) run first, then GPU-needing parts.

set -e
cd /p/scratch/training2625/zainzinger1/1337/recon_work/code
source /p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=/p/scratch/training2625/zainzinger1/1337/recon_work
export FL_DEVICE=cuda

echo "=== Stage 1: MLP SVD Projection (all 4 MLP models) ==="
python -u attack_mlp_svd_project.py model8 model1 model4 model5

echo ""
echo "=== Stage 2: ViT Analytic Inversion (model9, model11) ==="
python -u attack_vit_analytic.py model9 model11

echo ""
echo "=== Stage 3: Build a1 submission ==="
python -u build_a1.py

echo ""
echo "=== DONE ==="
echo "Outputs in: $FL_WORK_ROOT/outputs_a1/"
echo "Candidates in: $FL_WORK_ROOT/candidates_a1/"
