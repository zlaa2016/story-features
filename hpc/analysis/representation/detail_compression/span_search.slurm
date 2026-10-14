#!/usr/bin/env bash
#SBATCH --job-name=span_rep
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1

set -euo pipefail
source hpc/slurm_env.sh

# Array task 0 = contextual vectors, task 1 = isolated vectors. CPU only: the stored
# detail vectors are reused, nothing is re-extracted.
EXTRACTION="results/features/representation_extraction/causal_lm/detail_extraction"
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
  KIND=contextual
  VECTOR_DIR="${CONTEXT_DIR:-$EXTRACTION/${MODEL_KEY}_global_current}"
else
  KIND=isolated
  VECTOR_DIR="${ISOLATED_DIR:-$EXTRACTION/${MODEL_KEY}_detail_controls}"
fi
export KIND VECTOR_DIR
compgen -G "$VECTOR_DIR/*_detail_metadata_*.csv" >/dev/null || { echo "No detail metadata shards under $VECTOR_DIR" >&2; exit 1; }

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/detail_compression/span_search.py \
  --kind "$KIND" \
  --vector-dir "$VECTOR_DIR" \
  --pooling "${POOLING:-mean}" \
  --layers "${LAYERS:-}" \
  --max-span "${MAX_SPAN:-10}" \
  --length-penalties "${LENGTH_PENALTIES:-0}" \
  --limit "${LIMIT:-0}"'
