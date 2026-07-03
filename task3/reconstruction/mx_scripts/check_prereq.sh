#!/bin/bash
WORK=/p/scratch/training2625/zainzinger1/1337/recon_work
for m in model2 model3 model7; do
  echo "=== $m ==="
  ls -la $WORK/candidates/$m/ 2>&1 | grep -E "analytic_features|analytic_candidates|conv_invert" || echo "  (no analytic/conv files)"
done
echo "=== models.py conv_feature + stride ==="
grep -n "conv_feature\|self.stride\|self.padding\|feature_shape" $WORK/code/models.py | head -30
