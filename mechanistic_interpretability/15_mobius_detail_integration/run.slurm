#!/usr/bin/env bash
#SBATCH --job-name=detail_mobius
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-7

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

export SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.15_mobius_detail_integration.compute \
  --cohort mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl \
  --output-dir mechanistic_interpretability/outputs/15_mobius_detail_integration \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --shard-id "$(printf "%03d" "$SHARD_INDEX")"'
