#!/bin/bash
#SBATCH --job-name=nc_rscm_lamsweep
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=logs/lamsweep_%A_%a.out
#SBATCH --error=logs/lamsweep_%A_%a.err

set -euo pipefail

# λ sweep: 6 values in log space. Current CV-picked λ = 0.297 → eff_rank = 21.
# Goal: find λ that compresses eff_rank toward 3-5 while preserving k=3/5 pc_r².
LAMBDAS=(0.3 1.0 3.0 10.0 30.0 100.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

set +u
source ~/anaconda3/bin/activate
conda activate sfcoupling 2>/dev/null || conda activate base
set -u

cd "${REPO_ROOT:-$(pwd)}"

echo "=================================================================="
echo "λ-sweep task $SLURM_ARRAY_TASK_ID : λ=$LAM"
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  host $(hostname)"
echo "=================================================================="

OUT="results/rscm_ukb1079_lam_${LAM}"

python -u train/run_rscm.py \
    --config population_scale/configs/config_rscm_smoke_ukb1079_le_harmon.yaml \
    --seeds 42 43 44 \
    --d 53 \
    --rank_cap 50 \
    --pca_ks 3 5 10 20 \
    --lambda_grid $LAM \
    --out_dir $OUT

echo "DONE λ=$LAM at $(date)"
