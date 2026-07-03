#!/bin/bash
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
echo "=== recent output files ==="
ls -lat $WORK/output/ | head -12
echo "=== squeue all mine ==="
squeue -u zainzinger1 2>&1
echo "=== sacct via full path ==="
which sacct 2>&1
/usr/bin/sacct -X -j 15402804 --format=JobID,State,Elapsed,ExitCode,Reason 2>&1 || sacct -X -j 15402804 2>&1 || echo "sacct unavailable"
echo "=== any 154028xx out files ==="
ls -lat $WORK/output/ | grep -E "154028" | head
