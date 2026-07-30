#!/usr/bin/env bash
#SBATCH --job-name=amazon_search
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs data/amazon_data

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export INPUT_CSV="${INPUT_CSV:-data/processed/youtube_catalog.csv.gz}"
export OUTPUT_CSV="${OUTPUT_CSV:-data/amazon_data/amazon_from_hpc.csv}"
export MAX_BOOKS="${MAX_BOOKS:-}"
export SKIP_ROWS="${SKIP_ROWS:-0}"
export PENDING_ONLY="${PENDING_ONLY:-0}"
export STATUS_COLUMN="${STATUS_COLUMN:-ready_for_amazon_search}"
export SLEEP_SECS="${SLEEP_SECS:-5.0}"
export TIMEOUT_SECS="${TIMEOUT_SECS:-30}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   EXTRA_ARGS=(); \
   if [[ -n "${MAX_BOOKS:-}" ]]; then EXTRA_ARGS+=(--max-books "$MAX_BOOKS"); fi; \
   if [[ "${PENDING_ONLY:-0}" == "1" ]]; then EXTRA_ARGS+=(--pending-only); fi; \
   "$PYTHON_BIN" scripts/search_amazon_hpc.py \
      --input-csv "$INPUT_CSV" \
      --output-csv "$OUTPUT_CSV" \
      --skip-rows "$SKIP_ROWS" \
      --status-column "$STATUS_COLUMN" \
      --sleep "$SLEEP_SECS" \
      --timeout "$TIMEOUT_SECS" \
      "${EXTRA_ARGS[@]}"'
