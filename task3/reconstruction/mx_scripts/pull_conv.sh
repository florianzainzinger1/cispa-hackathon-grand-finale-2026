#!/bin/bash
DST="/mnt/c/Users/flori/AppData/Local/Temp/claude/C--Users-flori-OneDrive-Dokumente-hackathon-cispaFinale/3b687db5-ada4-4b68-b84b-77fa27b0a3e0/scratchpad"
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
scp -q -o BatchMode=yes "jureca:$WORK/candidates/model3/grid_conv_invert.png" "$DST/conv_model3_existing.png" && echo "got model3"
scp -q -o BatchMode=yes "jureca:$WORK/candidates/model7/grid_conv_invert.png" "$DST/conv_model7_existing.png" && echo "got model7"
