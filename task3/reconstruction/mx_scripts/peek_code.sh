#!/bin/bash
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
echo "===== validate_submission.py ====="
sed -n '1,120p' $WORK/code/validate_submission.py
echo "===== save_image_grid in utils.py ====="
grep -n "def save_image_grid\|def save_grid\|def montage\|save_image" $WORK/code/utils.py
