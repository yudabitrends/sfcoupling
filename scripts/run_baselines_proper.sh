#!/bin/bash
#SBATCH --job-name=baselines_proper
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/baselines_proper_%j.out
#SBATCH --error=logs/baselines_proper_%j.err

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
echo "Baselines Proper Training (Ridge + MLP 200ep)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Config: train/config_baselines.yaml"
echo "============================================="

python train/train_baselines.py \
    --config train/config_baselines.yaml \
    --exp_name baselines_proper

echo ""
echo "Baselines proper done at $(date)"
