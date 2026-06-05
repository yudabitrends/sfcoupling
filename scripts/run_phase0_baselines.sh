#!/bin/bash
#SBATCH --job-name=p0_baselines
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/p0_baselines_%j.out
#SBATCH --error=logs/p0_baselines_%j.err

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
echo "Phase 0.2: Multi-seed Baselines"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

python train/run_baselines_multiseed.py \
    --config train/config_baselines.yaml \
    --seeds 42 43 44 45 46 \
    --pca_ks 5 10 20 50 \
    --mlp_epochs 100 \
    --mlp_patience 15 \
    --out_dir results/baselines_multiseed

echo ""
echo "Phase 0.2 done at $(date)"
