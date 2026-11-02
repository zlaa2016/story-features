#!/usr/bin/env bash
#SBATCH --job-name=spectral_extract
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

# Vectors of the spectral-age batch videos (original stories only) for one model.  Array task = shard.
# STAGE=global  -> whole-story vectors + contextual detail vectors      (extract_global.py)
# STAGE=detail  -> isolated detail vectors from the same records         (extract_detail_controls.py)
# Submit through submit_extraction.sh, which sets the array size and chains the two stages.
set -euo pipefail
source hpc/slurm_env.sh

export MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
export STAGE="${STAGE:-global}"
export ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
export COHORT="${COHORT:-$ROOT/spectral_age_cohort.jsonl}"
export GLOBAL_DIR="${GLOBAL_DIR:-$ROOT/${MODEL_KEY}_spectral_global}"
export DETAIL_DIR="${DETAIL_DIR:-$ROOT/${MODEL_KEY}_spectral_detail_controls}"
export RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-100}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
export START_OFFSET="${START_OFFSET:-0}"     # first cohort record this submission is responsible for
export SHARD_OFFSET="${SHARD_OFFSET:-0}"     # shard ids start here, so a later submission never touches earlier shard files
export START="$((START_OFFSET + TASK * RECORDS_PER_SHARD))"
SHARD="$(printf "%03d" "$((SHARD_OFFSET + TASK))")"

if [[ "$STAGE" == "global" ]]; then
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/representation_extraction/causal_lm/detail_extraction/extract_global.py \
    --cohort "$COHORT" --model-key "$MODEL_KEY" --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
    --layers "${LAYERS:-0,4,8,12,16,20,-1}" --max-tokens "${MAX_TOKENS:-32768}" --record-types original \
    --overflow-policy "${OVERFLOW_POLICY:-skip}" --deletion-mode none \
    --start "$START" --limit "$RECORDS_PER_SHARD" --shard-id "'"$SHARD"'" --output-dir "$GLOBAL_DIR"'
else
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/representation_extraction/causal_lm/detail_extraction/extract_detail_controls.py \
    --cohort "$COHORT" --global-input-dir "$GLOBAL_DIR" --output-dir "$DETAIL_DIR" --model-key "$MODEL_KEY" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" --layers "${LAYERS:-0,4,8,12,16,20,-1}" --mode isolated \
    --record-types original --isolated-batch-size "${ISOLATED_BATCH_SIZE:-16}" --max-tokens "${MAX_TOKENS:-32768}" \
    --overflow-policy "${OVERFLOW_POLICY:-skip}" --start "$START" --limit "$RECORDS_PER_SHARD" \
    --shard-id "'"$SHARD"'" --records-from-global-shard'
fi
