#!/bin/bash
set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
LOCAL="/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon/task3/reconstruction"
# sync inverter + job
tr -d '\r' < "$LOCAL/mx_ssiminv.py" > /tmp/mx_ssiminv.py
scp -q -o BatchMode=yes /tmp/mx_ssiminv.py jureca:$WORK/code/mx_ssiminv.py
tr -d '\r' < "$LOCAL/mx_scripts/job_ssiminv.sh" > /tmp/job_ssiminv.sh
scp -q -o BatchMode=yes /tmp/job_ssiminv.sh jureca:$WORK/code/job_ssiminv.sh
ssh -o BatchMode=yes jureca "cd $WORK/code && sed -i 's/\r\$//' job_ssiminv.sh && sbatch job_ssiminv.sh"
