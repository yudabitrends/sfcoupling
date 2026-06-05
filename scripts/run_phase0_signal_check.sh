#!/bin/bash
#SBATCH --job-name=p0_signal
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/p0_signal_%j.out
#SBATCH --error=logs/p0_signal_%j.err

set -eo pipefail

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4

PROJECT_DIR=/home/users/ybi3/sfcoupling
cd "${PROJECT_DIR}"
mkdir -p logs

export PS1="${PS1:-}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "============================================="
echo "Phase 0.1: Signal Check"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

# Step 0.1: Check signal with multiple pca_k values
# Use a training config that has paths.aligned_features_dir
for PCA_K in 5 20 50; do
    echo ""
    echo "--- pca_k=${PCA_K} ---"
    python train/check_signal.py \
        --config train/config_baselines.yaml \
        --pca_k ${PCA_K} \
        --alpha 1.0 \
        --seed 42
done

echo ""
echo "Phase 0.1 done at $(date)"
