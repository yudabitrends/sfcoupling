#!/bin/bash
# Master launcher for 5-fold cross-validation pipeline.
# Usage: bash scripts/submit_cv.sh

set -euo pipefail

PROJECT_DIR=/home/users/ybi3/sfcoupling
cd "${PROJECT_DIR}"

echo "=== Step 1: Generate CV splits ==="
python preprocess/generate_cv_splits.py --K 5 --cv_seed 0

echo ""
echo "=== Step 2: Submit training array (15 jobs: 5 folds x 3 methods) ==="
JOB_ID=$(sbatch --parsable scripts/run_cv_array.sh)
echo "Submitted array job: ${JOB_ID}"

echo ""
echo "=== Step 3: Submit aggregation (depends on array completion) ==="
AGG_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} scripts/run_cv_aggregate.sh)
echo "Submitted aggregation job: ${AGG_ID}"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "  Training array: ${JOB_ID} (15 tasks)"
echo "  Aggregation:    ${AGG_ID} (runs after training completes)"
