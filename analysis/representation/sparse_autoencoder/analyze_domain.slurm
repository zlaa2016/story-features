#!/usr/bin/env bash
#SBATCH --job-name=sae_domain_eval
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/mechanistic_interpretability/sae_domain_eval-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_domain_eval-%A_%a.err

set -euo pipefail
source shared/slurm_env.sh
LAYERS=(8 16 24); LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
run_in_container "PYTHONPATH=. $PYTHON_BIN analysis/representation/sparse_autoencoder/analyze.py --checkpoint results/analysis/representation/sparse_autoencoder/checkpoints/domain_sae_layer_$(printf '%02d' "$LAYER").pt --output-dir results/analysis/representation/sparse_autoencoder/analysis/domain_trained/layer_$(printf '%02d' "$LAYER")"
