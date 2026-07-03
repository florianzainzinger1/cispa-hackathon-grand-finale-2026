#!/bin/bash
REM=/p/scratch/training2625/zainzinger1/1337/recon_work/reports/inspect_new
DST="/mnt/c/Users/flori/AppData/Local/Temp/claude/C--Users-flori-OneDrive-Dokumente-hackathon-cispaFinale/3b687db5-ada4-4b68-b84b-77fa27b0a3e0/scratchpad"
mkdir -p "$DST"
for f in "$@"; do
  scp -q -o BatchMode=yes "jureca:$REM/$f.png" "$DST/" && echo "got $f" || echo "FAIL $f"
done
ls -la "$DST"/*.png
