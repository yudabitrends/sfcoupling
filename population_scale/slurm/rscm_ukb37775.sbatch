#!/bin/bash
#SBATCH --job-name=nc_rscm_ukb37775
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/ukb37775_%j.out
#SBATCH --error=logs/ukb37775_%j.err

set -euo pipefail

set +u
source ~/anaconda3/bin/activate
conda activate sfcoupling 2>/dev/null || conda activate base
set -u

cd "${REPO_ROOT:-$(pwd)}"

echo "=================================================================="
echo "NC Phase 1 W3 (advanced): RSCM LE+harmon λ=0.3 on FULL UKB N=37,775"
echo "DS1=30,220  DS2=7,555  train=21,154"
echo "Job $SLURM_JOB_ID host $(hostname) date $(date)"
echo "=================================================================="

python -u train/run_rscm.py \
    --config population_scale/configs/config_rscm_ukb37775_le_harmon_lam03.yaml \
    --seeds 42 43 44 \
    --d 53 \
    --rank_cap 50 \
    --pca_ks 3 5 10 20 50 \
    --lambda_grid 0.3 \
    --save_decomposition \
    --out_dir results/rscm_ukb37775_le_harmon_lam03

echo "DONE at $(date)"
