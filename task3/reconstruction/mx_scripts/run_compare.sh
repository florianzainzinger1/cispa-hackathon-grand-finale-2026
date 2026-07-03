#!/bin/bash
set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
DATA=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
LOCAL="/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon/task3/reconstruction"
tr -d '\r' < "$LOCAL/mx_compare.py" > /tmp/mx_compare.py
scp -q -o BatchMode=yes /tmp/mx_compare.py jureca:$WORK/code/mx_compare.py
ssh -o BatchMode=yes jureca "source $DATA/.venv/bin/activate && export FL_WORK_ROOT=$WORK FL_DATA_ROOT=$DATA FL_DEVICE=cpu && cd $WORK/code && python -u mx_compare.py"
