#!/usr/bin/env bash
#SBATCH --job-name=sae_train
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/mechanistic_interpretability/sae_train-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_train-%A_%a.err

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh
LAYERS=(8 16 24); LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
run_in_container "PYTHONPATH=src $PYTHON_BIN src/analysis/representation/sparse_autoencoder/train.py --layer $LAYER --output results/analysis/representation/sparse_autoencoder/checkpoints/domain_sae_layer_$(printf '%02d' "$LAYER").pt"
