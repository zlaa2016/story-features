#!/usr/bin/env bash
#SBATCH --job-name=detail_align
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

INPUT_BASE="${DETAIL_INPUT_BASE:-mechanistic_interpretability/outputs/00_dataset_index}"
ROWS_PER_TASK="${ROWS_PER_TASK:-300}"
START=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))
OUTPUT_BASE="${DETAIL_OUTPUT_DIR:-mechanistic_interpretability/outputs/13_structured_detail_alignment}"
OUTPUT_DIR="${OUTPUT_BASE}/shard_${SLURM_ARRAY_TASK_ID}"

run_in_container '"$PYTHON_BIN" -m mechanistic_interpretability.13_structured_detail_alignment.align \
  --originals "'"$INPUT_BASE"'/originals.jsonl" \
  --summaries "'"$INPUT_BASE"'/summaries.jsonl" \
  --output-dir "'"$OUTPUT_DIR"'" \
  --model "${ALIGNMENT_MODEL:-sentence-transformers/all-mpnet-base-v2}" \
  --start "'"$START"'" --limit "'"$ROWS_PER_TASK"'"'
