#!/usr/bin/env bash
#SBATCH --job-name=gm_dfnc
#SBATCH --output=logs/gm_dfnc_%j.out
#SBATCH --error=logs/gm_dfnc_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/users/ybi3/sfcoupling

CONFIG_PATH="${1:-train/config_gm_dfnc_template.yaml}"

echo "[gm_dfnc] config: ${CONFIG_PATH}"
python scripts/run_gm_dfnc_analysis.py --config "${CONFIG_PATH}"
