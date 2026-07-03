set -e
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
DATA=/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction
source $DATA/.venv/bin/activate
export FL_WORK_ROOT=$WORK FL_DATA_ROOT=$DATA
echo "=== key files ==="
for f in outputs_final/test_vithi45.pt outputs_final/test_nlm05.pt outputs_trap/rec_model12.pt outputs_trap/rec_model10.pt gallery/celeba_top200_cc178.pt; do
  ls -la $WORK/$f 2>&1 || echo "MISSING $f"
done
echo "=== outputs_mx ==="
ls -la $WORK/outputs_mx/ 2>&1 || echo "no outputs_mx"
echo "=== reports/mx ==="
ls -la $WORK/reports/mx/ 2>&1 || echo "no reports/mx"
echo "=== reports/inspect_new ==="
ls -la $WORK/reports/inspect_new/ 2>&1 || echo "no inspect_new"
echo "=== code dir key files ==="
ls $WORK/code/ | grep -E "validate_submission|rec_lib|attack_conv_invert|models.py|utils.py" 2>&1
