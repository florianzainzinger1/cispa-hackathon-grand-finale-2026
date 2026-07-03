#!/bin/bash
# SLURM job: run the merged pipeline at CLUSTER quality (GPU + high conv-invert steps).
#
#   sbatch job.sh
#
# Produces submission.pt + viz/ in this folder. The heavy step is the conv-inversion
# of the 6 CNN models (256 candidates each); on GPU with MP_CONV_STEPS=3000 this is
# minutes. The MLP low-rank / analytic parts and the metric postproc are CPU-light.
#
#SBATCH --account=training2625
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=mp-merged
#SBATCH --output=work/%j.out
#SBATCH --error=work/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p work

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- quality knobs (override on the sbatch line if desired) ---
export MP_CONV_STEPS=${MP_CONV_STEPS:-3000}     # cluster: sharper full-res inversions
export MP_TOPK=${MP_TOPK:-256}

# --- data location: dir containing gradients/ and models/ ---
# defaults to the shared FL data on JURECA scratch; override with FL_DATA_ROOT=...
export FL_DATA_ROOT=${FL_DATA_ROOT:-/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction}

# --- python env with CUDA torch + scikit-image (see requirements.txt) ---
# point MP_VENV at a GPU venv; falls back to the repo venvs.
if [ -n "$MP_VENV" ] && [ -f "$MP_VENV/bin/activate" ]; then
    source "$MP_VENV/bin/activate"
elif [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"
srun python -u build.py --no-cache
echo "done -> $(pwd)/submission.pt"
