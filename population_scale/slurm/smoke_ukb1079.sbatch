#!/bin/bash
#SBATCH --job-name=nc_rscm_smoke_ukb1079
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_ukb1079_%j.out
#SBATCH --error=logs/smoke_ukb1079_%j.err

set -euo pipefail

# Conda activate with unbound-var workaround (known pattern in MEMORY)
set +u
source ~/anaconda3/bin/activate
conda activate sfcoupling 2>/dev/null || conda activate base
set -u

cd "${REPO_ROOT:-$(pwd)}"

echo "===================================================================="
echo "NC Phase 1.1 smoke: RSCM LE+harmon on UKB N=1,079"
echo "Job: $SLURM_JOB_ID  Host: $(hostname)  Date: $(date)"
echo "Python: $(which python)  $(python --version)"
echo "===================================================================="

python -u train/run_rscm.py \
    --config population_scale/configs/config_rscm_smoke_ukb1079_le_harmon.yaml \
    --seeds 42 43 44 \
    --d 53 \
    --rank_cap 50 \
    --pca_ks 5 10 20 50 \
    --save_decomposition \
    --out_dir results/rscm_smoke_ukb1079_le_harmon

echo "===================================================================="
echo "DONE at $(date)"
echo "===================================================================="
