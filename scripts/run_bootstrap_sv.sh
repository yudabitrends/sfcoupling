#!/bin/bash
#SBATCH --job-name=boot_sv
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/bootstrap_sv_%j.out
#SBATCH --error=logs/bootstrap_sv_%j.err

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
echo "Bootstrap SV Stability Analysis"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "============================================="

python train/run_bootstrap_sv.py \
    --config train/config_baselines.yaml \
    --n_boot 200 \
    --max_sv 30 \
    --seed 42 \
    --out_dir results/bootstrap_sv

echo ""
echo "Bootstrap SV analysis done at $(date)"
