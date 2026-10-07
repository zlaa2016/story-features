#!/usr/bin/env bash
#SBATCH --job-name=mech_generate
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-12

set -euo pipefail
source hpc/slurm_env.sh

export STORIES_PER_SHARD="${STORIES_PER_SHARD:-300}"
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export START="$((SHARD_ID * STORIES_PER_SHARD))"
export MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export OUTPUT="results/features/summaries/native_generation/${MODEL_KEY}/generations_$(printf "%03d" "$SHARD_ID").jsonl"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/summaries/native_generation/generate.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --model-key "$MODEL_KEY" \
  --start "$START" \
  --limit "$STORIES_PER_SHARD" \
  --n-summaries "${N_SUMMARIES:-1}" \
  --available-conditions-only \
  --compression-ratio 0.30 \
  --output "$OUTPUT"'
