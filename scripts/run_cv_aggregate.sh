#!/bin/bash
#SBATCH --job-name=cv_agg
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/cv_agg_%j.out
#SBATCH --error=logs/cv_agg_%j.err

set -eo pipefail

PROJECT_DIR=/home/users/ybi3/sfcoupling
cd "${PROJECT_DIR}"
mkdir -p logs

export PS1="${PS1:-}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "============================================="
echo "Aggregating CV results"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================="

python scripts/aggregate_cv_results.py --results_dir results/cv --K 5

echo ""
echo "CV aggregation done at $(date)"
