#!/usr/bin/env bash
#SBATCH --job-name=detail_chars
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.natural_language.detail_characteristics.extract \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --output-dir results/analysis/natural_language/detail_characteristics \
  --retention-csv results/analysis/natural_language/detail_alignment/original_detail_retention.csv && \
PYTHONPATH=src "$PYTHON_BIN" -m analysis.natural_language.detail_characteristics.analyze \
  --input-dir results/analysis/natural_language/detail_characteristics \
  --retention-csv results/analysis/natural_language/detail_alignment/original_detail_retention.csv'
