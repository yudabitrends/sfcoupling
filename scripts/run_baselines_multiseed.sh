#!/bin/bash
#SBATCH --job-name=bl_mseed
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/bl_mseed_%j.out
#SBATCH --error=logs/bl_mseed_%j.err

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
echo "Baselines Multi-seed (Ridge + MLP, 7 seeds)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

python train/run_baselines_multiseed.py \
    --config train/config_baselines.yaml \
    --seeds 42 43 44 45 46 47 48 \
    --pca_ks 5 10 20 50 \
    --mlp_epochs 200 \
    --mlp_patience 20 \
    --out_dir results/baselines_multiseed

echo ""
echo "Baselines multi-seed done at $(date)"
