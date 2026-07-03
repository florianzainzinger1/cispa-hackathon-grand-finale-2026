#!/bin/bash
JID=$1
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
for i in $(seq 1 60); do
  ST=$(squeue -j $JID -h -o "%T" 2>/dev/null)
  if [ -z "$ST" ]; then
    echo "job $JID no longer in queue (done or failed)"
    break
  fi
  echo "[poll $i] job $JID state=$ST"
  sleep 15
done
echo "===== OUT tail ====="
tail -40 $WORK/output/$JID.out 2>&1 || echo "no out"
echo "===== ERR tail ====="
tail -20 $WORK/output/$JID.err 2>&1 || echo "no err"
echo "===== sacct ====="
sacct -X -j $JID --format=JobID,State,Elapsed,ExitCode 2>&1 || true
