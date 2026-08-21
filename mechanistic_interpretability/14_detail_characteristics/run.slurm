#!/usr/bin/env bash
#SBATCH --job-name=detail_chars
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.14_detail_characteristics.extract \
  --cohort mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl \
  --output-dir mechanistic_interpretability/outputs/14_detail_characteristics \
  --retention-csv mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv && \
PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.14_detail_characteristics.analyze \
  --input-dir mechanistic_interpretability/outputs/14_detail_characteristics \
  --retention-csv mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv'
