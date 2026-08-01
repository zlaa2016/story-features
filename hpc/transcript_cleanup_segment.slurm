#!/usr/bin/env bash
#SBATCH --job-name=story_cleanseg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

set -euo pipefail

mkdir -p logs pipeline_output

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export INPUT_DIR="${INPUT_DIR:-data/labcomputer_output}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-transcript_cleanup_segment}"
export OUT_DIR="${OUT_DIR:-}"
export KEY_PATH="${KEY_PATH:-/scratch/$USER/.cache/openai/hartley_api_key.txt}"
export CLEAN_MODEL="${CLEAN_MODEL:-gpt-5.4-nano}"
export SEGMENT_MODEL="${SEGMENT_MODEL:-gpt-5.4-mini}"
export START_ROW="${START_ROW:-0}"
export LIMIT_ROWS="${LIMIT_ROWS:-}"
export MAX_CALLS="${MAX_CALLS:-}"
export SLEEP_SECS="${SLEEP_SECS:-0.25}"
export TRANSCRIPT_SOURCE="${TRANSCRIPT_SOURCE:-all}"
export MIN_WORDS="${MIN_WORDS:-20}"
export OVERWRITE="${OVERWRITE:-0}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   EXTRA_ARGS=(); \
   if [[ -n "${OUT_DIR:-}" ]]; then EXTRA_ARGS+=(--out-dir "$OUT_DIR"); fi; \
   if [[ -n "${LIMIT_ROWS:-}" ]]; then EXTRA_ARGS+=(--limit "$LIMIT_ROWS"); fi; \
   if [[ -n "${MAX_CALLS:-}" ]]; then EXTRA_ARGS+=(--max-calls "$MAX_CALLS"); fi; \
   if [[ "${OVERWRITE:-0}" == "1" ]]; then EXTRA_ARGS+=(--overwrite); fi; \
   "$PYTHON_BIN" scripts/run_transcript_cleanup_segmentation.py \
      --input-dir "$INPUT_DIR" \
      --output-prefix "$OUTPUT_PREFIX" \
      --key-path "$KEY_PATH" \
      --clean-model "$CLEAN_MODEL" \
      --segment-model "$SEGMENT_MODEL" \
      --start "$START_ROW" \
      --sleep "$SLEEP_SECS" \
      --transcript-source "$TRANSCRIPT_SOURCE" \
      --min-words "$MIN_WORDS" \
      "${EXTRA_ARGS[@]}"'
