#!/bin/bash
#SBATCH --account=training2625
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --reservation=cispahack --cpus-per-task=8 --partition=dc-gpu
#SBATCH --time=00:25:00
#SBATCH --output=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.out
#SBATCH --error=/p/scratch/training2625/zainzinger1/1337/recon_work/output/%j.err
TEAM=/p/scratch/training2625/zainzinger1/1337
cd $TEAM/recon_work/code
source $TEAM/FL_Data_Reconstruction/.venv/bin/activate
export FL_WORK_ROOT=$TEAM/recon_work FL_DATA_ROOT=$TEAM/FL_Data_Reconstruction
export FL_DEVICE=cuda REC_STEPS=${REC_STEPS:-900}
python -u rec_reconstruct.py ${REC_MODELS:-model12 model6 model2 model7 model3}
echo "JOB DONE"
