#!/usr/bin/env bash
#SBATCH --job-name=sae_official
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/mechanistic_interpretability/sae_official-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_official-%A_%a.err

set -euo pipefail
source shared/slurm_env.sh
LAYERS=(8 16 24); LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
run_in_container "PYTHONPATH=. $PYTHON_BIN analysis/representation/sparse_autoencoder/prepare_official.py --layer $LAYER --output results/analysis/representation/sparse_autoencoder/checkpoints/official_sae_layer_$(printf '%02d' "$LAYER").pt"
