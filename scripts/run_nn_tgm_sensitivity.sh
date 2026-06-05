#!/bin/bash
#SBATCH --job-name=nn_tgm_sens
#SBATCH --partition=qTRDGPU
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/nn_tgm_sens_%j.out
#SBATCH --error=logs/nn_tgm_sens_%j.err

# Nuclear Norm sensitivity analysis on TGM-residualized data
# The aligned features at /data/users1/ybi3/cVAE/aligned_features/ already
# include total_gm residualization (covariates: Age, Gender, total_gm).

set -eo pipefail
cd /home/users/ybi3/sfcoupling
mkdir -p logs results/multivariate_methods_tgm

set +u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
set -u

echo "=== Nuclear Norm TGM Sensitivity ==="
echo "Start: $(date)"

python train/run_multivariate_methods.py \
    --config train/config_baselines.yaml \
    --seeds 42 43 44 45 46 47 48 \
    --max_rank 30 \
    --pca_ks 5 10 20 50 \
    --n_perm 1000 \
    --n_boot 10000 \
    --save_decomposition \
    --out_dir results/multivariate_methods_tgm

echo "=== Subspace analysis ==="
python train/run_subspace_analysis.py \
    --config train/config_baselines.yaml \
    --methods_dir results/multivariate_methods_tgm \
    --out_dir results/subspace_analysis_tgm

echo "Done: $(date)"
