#!/usr/bin/env bash
set -euo pipefail

# Submit one Slurm job per model key in perplexity/models.py.
# By default this runs the core set. Set COMPARISON_SET to run another named comparison.
# Set MIN_CONTEXT_LENGTH=32768 for long-context-only runs.
# Run from the repository root:
#   bash perplexity/submit_all_perplexity.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
COMPARISON_SET="${COMPARISON_SET:-core}"
MIN_CONTEXT_LENGTH="${MIN_CONTEXT_LENGTH:-0}"
MODEL_KEYS="${MODEL_KEYS:-}"

if [[ -z "$MODEL_KEYS" ]]; then
  MODEL_ARGS=(--list-keys --min-context-length "$MIN_CONTEXT_LENGTH")
  if [[ -n "$COMPARISON_SET" ]]; then
    MODEL_ARGS+=(--comparison-set "$COMPARISON_SET")
  fi
  MODEL_KEYS="$("$PYTHON_BIN" -m perplexity.models "${MODEL_ARGS[@]}")"
fi

for key in $MODEL_KEYS; do
  echo "Submitting perplexity job for ${key}"
  sbatch --export=ALL,MODEL_KEY="${key}" perplexity/run_model_perplexity.slurm
done
