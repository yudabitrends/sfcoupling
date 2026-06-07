#!/bin/bash
#SBATCH --job-name=nc_lamsweep37k
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/ukb37775_lamsweep_%j.out
#SBATCH --error=logs/ukb37775_lamsweep_%j.err

set -euo pipefail

set +u
source ~/anaconda3/bin/activate
conda activate sfcoupling 2>/dev/null || conda activate base
set -u

cd "${REPO_ROOT:-$(pwd)}"

echo "=================================================================="
echo "B2: λ-sweep on UKB Tier 3 (N=37,775) addresses M2 audit finding"
echo "λ ∈ {0.1, 0.2, 0.3, 0.5, 1.0}, val_mse selection, single seed=42"
echo "Job $SLURM_JOB_ID host $(hostname) date $(date)"
echo "=================================================================="

python -u train/run_rscm.py \
    --config population_scale/configs/config_rscm_ukb37775_le_harmon_lam03.yaml \
    --seeds 42 \
    --d 53 \
    --rank_cap 50 \
    --pca_ks 3 5 10 20 50 \
    --lambda_grid 0.1 0.2 0.3 0.5 1.0 \
    --save_decomposition \
    --out_dir results/rscm_ukb37775_lamsweep

echo "DONE at $(date)"
