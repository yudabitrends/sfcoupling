#!/bin/bash
#SBATCH --job-name=nn_mlp_2stage
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/nn_mlp_2stage_%j.out
#SBATCH --error=logs/nn_mlp_2stage_%j.err

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
echo "Two-Stage Nuclear Norm + MLP (Direction C)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

python train/run_nn_mlp_twostage.py \
    --config train/config_baselines.yaml \
    --seeds 42 43 44 45 46 47 48 \
    --pca_ks 5 7 10 20 \
    --mlp_epochs 200 \
    --mlp_patience 20 \
    --n_boot 10000 \
    --out_dir results/nn_mlp_twostage

echo ""
echo "Two-stage NN+MLP done at $(date)"
