#!/usr/bin/env bash
#SBATCH --job-name=detail_align_stats
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh

OUTPUT_DIR="${DETAIL_OUTPUT_DIR:-mechanistic_interpretability/outputs/13_structured_detail_alignment}"

run_in_container '"$PYTHON_BIN" -m analysis.natural_language.detail_alignment.merge_shards \
  --input-dir "'"$OUTPUT_DIR"'"'
run_in_container '"$PYTHON_BIN" -m analysis.natural_language.detail_alignment.validate \
  --input-dir "'"$OUTPUT_DIR"'"'
run_in_container '"$PYTHON_BIN" -m analysis.natural_language.detail_alignment.analyze \
  --input-dir "'"$OUTPUT_DIR"'"'
