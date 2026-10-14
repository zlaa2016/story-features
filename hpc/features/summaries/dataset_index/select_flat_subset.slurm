#!/usr/bin/env bash
#SBATCH --job-name=flat_subset
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
source hpc/slurm_env.sh

# Stories with younger, actual and older summaries, flat across source age.
# Override with PER_BIN=<n> (stories per age bin) or NO_FLATTEN=1 to keep all complete stories.
# Sentence-level segmentation (local_sentence) is excluded by default so every summary unit is a detail;
# set EXCLUDE_SEGMENTATION_METHODS="" to keep it, or SEGMENTATION_METHOD=api_detail to keep only that method.
run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/summaries/dataset_index/select_flat_subset.py \
  --cohort-metadata results/features/summaries/dataset_index/cohort_metadata.csv \
  --output results/features/summaries/dataset_index/flat_subset.csv \
  --per-bin "${PER_BIN:-0}" --min-bin "${MIN_BIN:-5}" --require-method "${SEGMENTATION_METHOD:-}" --exclude-methods "${EXCLUDE_SEGMENTATION_METHODS-local_sentence}" ${NO_FLATTEN:+--no-flatten}'
