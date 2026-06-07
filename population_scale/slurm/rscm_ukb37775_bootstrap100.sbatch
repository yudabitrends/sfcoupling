#!/bin/bash
#SBATCH --job-name=nc_boot100_37k
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/ukb37775_boot100_%j.out
#SBATCH --error=logs/ukb37775_boot100_%j.err

set -euo pipefail

set +u
source ~/anaconda3/bin/activate
conda activate sfcoupling 2>/dev/null || conda activate base
set -u

cd "${REPO_ROOT:-$(pwd)}"

echo "=================================================================="
echo "B1: 100-bootstrap CI on UKB Tier 3 (N=37,775) — replaces multi-seed CI"
echo "Estimated wall-clock: ≈ 6 GPU-hr (100 × ~215 s per fit)"
echo "Job $SLURM_JOB_ID host $(hostname) date $(date)"
echo "=================================================================="

python -u scripts/run_bootstrap_tier3.py \
    --config population_scale/configs/config_rscm_ukb37775_le_harmon_lam03.yaml \
    --n_bootstraps 100 \
    --lambda_fixed 0.3 \
    --rank_cap 50 \
    --reference_U results/rscm_ukb37775_le_harmon_lam03/decompositions/rscm_seed42_U.npy \
    --out_dir results/rscm_ukb37775_bootstrap100

echo "DONE at $(date)"
