#!/usr/bin/env bash
#SBATCH --job-name=global_repr_analysis
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

export MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export INPUT_DIR="${INPUT_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_KEY}_global}"
export OUTPUT_DIR="${OUTPUT_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_KEY}_global_analysis}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" mechanistic_interpretability/02_representation_extraction/analyze_global.py \
  --input-dir "$INPUT_DIR" \
  ${DELETION_INPUT_DIR:+--deletion-input-dir "$DELETION_INPUT_DIR"} \
  --output-dir "$OUTPUT_DIR"'
