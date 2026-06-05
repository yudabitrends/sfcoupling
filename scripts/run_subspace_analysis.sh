#!/bin/bash
#SBATCH --job-name=subspace_analysis
#SBATCH --partition=qTRDGPUL
#SBATCH --account=trends53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/subspace_analysis_%j.out
#SBATCH --error=logs/subspace_analysis_%j.err

set -eo pipefail

cd /home/users/ybi3/sfcoupling
mkdir -p logs

# Conda activation (unbound $PS1 workaround)
set +u
source ~/anaconda3/bin/activate
conda activate base
set -u

echo "=== Subspace Analysis ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo ""

python train/run_subspace_analysis.py \
    --config train/config_baselines.yaml \
    --seed 42 43 44 45 46 47 48 \
    --subspace_ks 5 10 20 30 \
    --n_perm 1000 \
    --out_dir results/subspace_analysis

echo ""
echo "=== Done: $(date) ==="
