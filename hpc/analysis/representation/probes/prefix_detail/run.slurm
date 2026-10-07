#!/usr/bin/env bash
#SBATCH --job-name=prefix_detail_probe
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

set -euo pipefail
source hpc/slurm_env.sh
export FAKEROOT="${FAKEROOT:-0}"

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
PREFIX_MODE="${PREFIX_MODE:-native}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-5}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
START="$((SHARD_ID * RECORDS_PER_SHARD))"
OUTPUT_DIR="${OUTPUT_DIR:-mechanistic_interpretability/outputs/18_prefix_detail_probes/${MODEL_KEY}_${PREFIX_MODE}}"
SAVE_TOKEN_STATES_ARG=""
if [[ "${SAVE_TOKEN_STATES:-0}" == "1" ]]; then
  SAVE_TOKEN_STATES_ARG="--save-token-states"
fi
export MODEL_KEY PREFIX_MODE RECORDS_PER_SHARD SHARD_ID START OUTPUT_DIR SAVE_TOKEN_STATES_ARG

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/probes/prefix_detail/extract.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --model-key "$MODEL_KEY" \
  --prefix-mode "$PREFIX_MODE" \
  --prefix-step "${PREFIX_STEP:-1}" \
  --conditions "${CONDITIONS:-isolated,true}" \
  --permutations "${PERMUTATIONS:-1}" \
  --sample-stories "${SAMPLE_STORIES:-0}" \
  --sample-seed "${SAMPLE_SEED:-2026}" \
  --layers "${LAYERS:-0,6,12,18,-1}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --max-details-per-story "${MAX_DETAILS_PER_STORY:-0}" \
  --start "$START" \
  --limit "$RECORDS_PER_SHARD" \
  --shard-id "$(printf "%04d" "$SHARD_ID")" \
  --output-dir "$OUTPUT_DIR" \
  ${SAVE_TOKEN_STATES_ARG:-}'
