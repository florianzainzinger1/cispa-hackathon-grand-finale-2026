WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
echo "===== reports/mx/STATUS.md ====="
cat $WORK/reports/mx/STATUS.md 2>&1 || echo "no status"
echo ""
echo "===== code dir mx_*.py ====="
ls -la $WORK/code/mx_*.py 2>&1 || echo "no mx code"
