#!/bin/bash
#SBATCH --job-name=cv_sfcoupling
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/cv_%A_%a.out
#SBATCH --error=logs/cv_%A_%a.err

set -eo pipefail

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4

PROJECT_DIR=/home/users/ybi3/sfcoupling
cd "${PROJECT_DIR}"
mkdir -p logs

export PS1="${PS1:-}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

# Decode array index: 5 folds x 3 scripts
FOLD=$(( SLURM_ARRAY_TASK_ID / 3 ))
SCRIPT_ID=$(( SLURM_ARRAY_TASK_ID % 3 ))

SEEDS="42 43 44 45 46 47 48"
CONFIG="train/config_cv_fold_${FOLD}.yaml"

echo "============================================="
echo "CV Array Job: fold=${FOLD}, script=${SCRIPT_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
echo "============================================="

case ${SCRIPT_ID} in
    0)
        echo "Running baselines (Ridge + MLP)..."
        python train/run_baselines_multiseed.py \
            --config "${CONFIG}" \
            --seeds ${SEEDS} \
            --out_dir "results/cv/fold_${FOLD}/baselines_multiseed"
        ;;
    1)
        echo "Running multivariate methods (RRR + PLS + NucNorm)..."
        python train/run_multivariate_methods.py \
            --config "${CONFIG}" \
            --seeds ${SEEDS} \
            --n_perm 0 \
            --n_boot 0 \
            --out_dir "results/cv/fold_${FOLD}/multivariate_methods"
        ;;
    2)
        echo "Running kernel spectral regression..."
        python train/run_kernel_spectral_regression.py \
            --config "${CONFIG}" \
            --seeds ${SEEDS} \
            --n_perm 0 \
            --n_boot 0 \
            --out_dir "results/cv/fold_${FOLD}/kernel_spectral_regression"
        ;;
esac

echo ""
echo "CV fold ${FOLD} script ${SCRIPT_ID} done at $(date)"
