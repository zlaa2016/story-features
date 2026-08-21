#!/usr/bin/env bash
#SBATCH --job-name=sae_analyze
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/mechanistic_interpretability/sae_analyze-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_analyze-%A_%a.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
LAYERS=(8 16 24); LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
run_in_container "PYTHONPATH=. $PYTHON_BIN mechanistic_interpretability/12_sparse_autoencoder/analyze.py --checkpoint mechanistic_interpretability/outputs/12_sparse_autoencoder/checkpoints/official_sae_layer_$(printf '%02d' "$LAYER").pt --output-dir mechanistic_interpretability/outputs/12_sparse_autoencoder/analysis/official/layer_$(printf '%02d' "$LAYER")"
