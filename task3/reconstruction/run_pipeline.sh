#!/bin/bash
# One-command pipeline (login-node stages are CPU-light; ViT optimisation is a
# separate GPU sbatch: job_recon.sh). Run from recon_work/code with the FL venv
# active and FL_WORK_ROOT set.
set -e
echo "== 1. inspect ==";        python -u inspect_models.py
echo "== 2. analytic ==";       python -u attack_analytic.py
echo "== 3. conv-invert ==";    python -u attack_conv_invert.py
echo "== 4. label hints ==";    python -u infer_labels.py
echo "== 5. postprocess ==";    python -u postprocess.py
echo "== 6. face bank ==";      python -u face_bank.py
echo "== 7. build submission =="; python -u build_submission.py
echo "== 8. validate ==";       python -u validate_submission.py
echo "pipeline done"
