#!/bin/bash
#SBATCH --job-name=mv_methods
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/mv_methods_%j.out
#SBATCH --error=logs/mv_methods_%j.err

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
echo "Multivariate Methods: RRR + PLS + Nuclear Norm"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

python train/run_multivariate_methods.py \
    --config train/config_baselines.yaml \
    --seeds 42 43 44 45 46 47 48 \
    --max_rank 30 \
    --pca_ks 5 10 20 50 \
    --n_perm 1000 \
    --n_boot 10000 \
    --save_decomposition \
    --out_dir results/multivariate_methods

echo ""
echo "Multivariate methods done at $(date)"
