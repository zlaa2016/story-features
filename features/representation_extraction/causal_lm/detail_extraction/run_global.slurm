#!/usr/bin/env bash
#SBATCH --job-name=global_repr
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-39%4

set -euo pipefail
source shared/slurm_env.sh

export RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-250}"
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export START="$((SHARD_ID * RECORDS_PER_SHARD))"
export MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export OUTPUT_DIR="${OUTPUT_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_KEY}_global}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" features/representation_extraction/causal_lm/detail_extraction/extract_global.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --model-key "$MODEL_KEY" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --layers "${LAYERS:-0,4,8,12,16,20,-1}" \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --record-types "${RECORD_TYPES:-original,summary}" \
  --overflow-policy "${OVERFLOW_POLICY:-error}" \
  --deletion-mode "${DELETION_MODE:-none}" \
  --start "$START" \
  --limit "$RECORDS_PER_SHARD" \
  --shard-id "$(printf "%03d" "$SHARD_ID")" \
  --output-dir "$OUTPUT_DIR"'
