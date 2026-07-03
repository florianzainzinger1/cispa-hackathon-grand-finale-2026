#!/bin/bash
# runs ON jureca
JID=15402804
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
for i in $(seq 1 40); do
  ST=$(/usr/bin/sacct -X -j $JID -n -o State 2>/dev/null | head -1 | tr -d ' ')
  echo "[poll $i] $JID state=$ST"
  case "$ST" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY) break;;
  esac
  sleep 15
done
echo "===== OUT ====="
cat $WORK/output/$JID.out 2>&1 || echo "no out"
echo "===== ERR tail ====="
tail -25 $WORK/output/$JID.err 2>&1 || echo "no err"
