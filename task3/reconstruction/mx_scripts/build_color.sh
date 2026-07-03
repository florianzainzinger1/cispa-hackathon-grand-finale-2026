#!/bin/bash
set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
DATA=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
LOCAL="/mnt/c/Users/flori/OneDrive/Dokumente/hackathon/cispaFinale/CISPA-Hackathon/task3/reconstruction"
# sync my mx_*.py files (strip CRLF)
for f in mx_colorfix.py mx_montage.py; do
  tr -d '\r' < "$LOCAL/$f" > /tmp/$f
  scp -q -o BatchMode=yes /tmp/$f jureca:$WORK/code/$f
done
ssh -o BatchMode=yes jureca "bash -s" <<'REMOTE'
set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
DATA=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
source $DATA/.venv/bin/activate
export FL_WORK_ROOT=$WORK FL_DATA_ROOT=$DATA FL_DEVICE=cpu
cd $WORK/code
echo "=== building color candidates ==="
python -u mx_colorfix.py
echo "=== validating ==="
for c in cand_m12_color1 cand_m12_color2 cand_m12_color3 cand_m10_clean1 cand_m10_clean2; do
  echo "--- $c ---"
  python -u validate_submission.py $WORK/outputs_mx/$c.pt 2>&1 | tail -2
done
echo "=== montages ==="
python -u mx_montage.py outputs_mx/cand_m12_color1.pt model12 reports/inspect_new/cand_m12_color1.png
python -u mx_montage.py outputs_mx/cand_m12_color2.pt model12 reports/inspect_new/cand_m12_color2.png
python -u mx_montage.py outputs_mx/cand_m12_color3.pt model12 reports/inspect_new/cand_m12_color3.png
python -u mx_montage.py outputs_mx/cand_m10_clean1.pt model10 reports/inspect_new/cand_m10_clean1.png
python -u mx_montage.py outputs_mx/cand_m10_clean2.pt model10 reports/inspect_new/cand_m10_clean2.png
echo "ALL DONE"
REMOTE
