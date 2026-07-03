#!/bin/bash
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
echo "############ attack_conv_invert.py ############"
cat $WORK/code/attack_conv_invert.py
echo ""
echo "############ rec_lib.py (signatures + conv/feature helpers) ############"
grep -n "^def \|^class \|def invert\|feature\|fc1\|conv\|exact\|analytic" $WORK/code/rec_lib.py | head -80
