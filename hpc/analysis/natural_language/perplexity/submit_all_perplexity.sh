#!/usr/bin/env bash
set -euo pipefail

# Submit one Slurm job per model key in src/analysis/natural_language/perplexity/models.py.
# By default this runs every registered model.
# Set MIN_CONTEXT_LENGTH=32768 for long-context-only runs.
# Run from the repository root:
#   bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCOUNT="${ACCOUNT:-torch_pr_494_general}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-24G}"
MIN_CONTEXT_LENGTH="${MIN_CONTEXT_LENGTH:-0}"
MODEL_KEYS="${MODEL_KEYS:-}"

if [[ -z "$MODEL_KEYS" ]]; then
  MODEL_KEYS="$("$PYTHON_BIN" -m src/analysis/natural_language/perplexity.models --list-keys --min-context-length "$MIN_CONTEXT_LENGTH")"
fi

for key in $MODEL_KEYS; do
  echo "Submitting src/analysis/natural_language/perplexity job for ${key}"
  sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem="${MEM}" \
    --export=ALL,MODEL_KEY="${key}" \
    hpc/analysis/natural_language/perplexity/run_model_perplexity.slurm
done
