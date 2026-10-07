#!/usr/bin/env bash
#SBATCH --job-name=span_nl
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
source hpc/slurm_env.sh

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/natural_language/detail_compression/span_search.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --subset "${SUBSET:-results/features/summaries/dataset_index/flat_subset.csv}" \
  --output-dir "${OUTPUT_DIR:-results/analysis/natural_language/detail_compression/span_search}" \
  --model "${ALIGNMENT_MODEL:-sentence-transformers/all-mpnet-base-v2}" \
  --max-span "${MAX_SPAN:-10}" \
  --length-penalties "${LENGTH_PENALTIES:-0,0.1,0.2}" \
  --limit "${LIMIT:-0}"'
