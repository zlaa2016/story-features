#!/usr/bin/env bash
#SBATCH --job-name=sae_collect
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=logs/mechanistic_interpretability/sae_collect-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_collect-%A_%a.err

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh
LAYERS=(8 16 24); TASK="$SLURM_ARRAY_TASK_ID"; LAYER_INDEX=$((TASK / 10)); SHARD=$((TASK % 10)); LAYER="${LAYERS[$LAYER_INDEX]}"; START=$((SHARD * 1000))
run_in_container "PYTHONPATH=src $PYTHON_BIN src/analysis/representation/sparse_autoencoder/collect_activations.py --layer $LAYER --start $START --limit 1000 --shard-id $(printf '%02d' "$SHARD")"
