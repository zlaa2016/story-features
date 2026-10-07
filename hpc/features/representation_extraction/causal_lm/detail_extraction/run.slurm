#!/usr/bin/env bash
#SBATCH --job-name=mech_extract
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-14%4

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh

export RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-700}"
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export START="$((SHARD_ID * RECORDS_PER_SHARD))"
export MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export OUTPUT_DIR="results/features/representation_extraction/causal_lm/detail_extraction/${MODEL_KEY}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/representation_extraction/causal_lm/detail_extraction/extract.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --model-key "$MODEL_KEY" \
  --layers "${LAYERS:-0,4,8,12,16,20,-1}" \
  --start "$START" \
  --limit "$RECORDS_PER_SHARD" \
  --shard-id "$(printf "%03d" "$SHARD_ID")" \
  --output-dir "$OUTPUT_DIR"'
