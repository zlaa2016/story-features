#!/usr/bin/env bash
#SBATCH --job-name=causal_geom_align
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
source shared/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
INPUT_DIR="results/analysis/representation/geometry/compression_geometry/${MODEL_KEY}/records"
OUTPUT_DIR="results/analysis/representation/geometry/compression_geometry/${MODEL_KEY}/alignment/shard_${SLURM_ARRAY_TASK_ID}"
ROWS_PER_TASK="${ROWS_PER_TASK:-400}"
START=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m analysis.natural_language.detail_alignment.align \
  --originals "'"$INPUT_DIR"'/originals.jsonl" \
  --summaries "'"$INPUT_DIR"'/summaries.jsonl" \
  --output-dir "'"$OUTPUT_DIR"'" \
  --start "'"$START"'" \
  --limit "'"$ROWS_PER_TASK"'"'
