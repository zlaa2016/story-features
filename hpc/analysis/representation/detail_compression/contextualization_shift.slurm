#!/usr/bin/env bash
#SBATCH --job-name=ctx_shift
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
source hpc/slurm_env.sh

# Contextual-vs-isolated shift for original and summary details at every stored layer.
# CPU only: reuses the stored vectors. Same subset as the span search.
EXTRACTION="results/features/representation_extraction/causal_lm/detail_extraction"
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
export CONTEXT_DIR="${CONTEXT_DIR:-$EXTRACTION/${MODEL_KEY}_global_current}"
export ISOLATED_DIR="${ISOLATED_DIR:-$EXTRACTION/${MODEL_KEY}_detail_controls}"
for dir in "$CONTEXT_DIR" "$ISOLATED_DIR"; do
  compgen -G "$dir/*_detail_metadata_*.csv" >/dev/null || { echo "No detail metadata shards under $dir" >&2; exit 1; }
done

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/detail_compression/contextualization_shift.py \
  --context-dir "$CONTEXT_DIR" \
  --isolated-dir "$ISOLATED_DIR" \
  --subset "${SUBSET:-results/features/summaries/dataset_index/flat_subset.csv}" \
  --layers "${LAYERS:-}" \
  --limit "${LIMIT:-0}"'
