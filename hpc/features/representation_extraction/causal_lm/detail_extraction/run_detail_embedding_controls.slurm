#!/usr/bin/env bash
#SBATCH --job-name=detail_controls
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --array=0-169%4

set -euo pipefail
source hpc/slurm_env.sh

export RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-50}"
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
export START="$((SHARD_ID * RECORDS_PER_SHARD))"
export MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export MODE="${MODE:-isolated}"
export GLOBAL_INPUT_DIR="${GLOBAL_INPUT_DIR:-results/features/representation_extraction/causal_lm/detail_extraction/gpt_oss_20b_global_current}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/features/representation_extraction/causal_lm/detail_extraction/gpt_oss_20b_detail_controls}"
export RECORD_SELECTION_ARG=""
if [[ "${RECORDS_FROM_GLOBAL_SHARD:-0}" == "1" ]]; then
  export RECORD_SELECTION_ARG="--records-from-global-shard"
fi

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/representation_extraction/causal_lm/detail_extraction/extract_detail_controls.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --global-input-dir "$GLOBAL_INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-key "$MODEL_KEY" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --layers "${LAYERS:-0,4,8,12,16,20,-1}" \
  --mode "$MODE" \
  --record-types "${RECORD_TYPES:-original,summary}" \
  --deletion-record-types "${DELETION_RECORD_TYPES:-original}" \
  --max-deletions-per-record "${MAX_DELETIONS_PER_RECORD:-0}" \
  --isolated-batch-size "${ISOLATED_BATCH_SIZE:-16}" \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --overflow-policy "${OVERFLOW_POLICY:-error}" \
  --start "$START" \
  --limit "$RECORDS_PER_SHARD" \
  --shard-id "$(printf "%03d" "$SHARD_ID")" \
  ${RECORD_SELECTION_ARG:-}'
