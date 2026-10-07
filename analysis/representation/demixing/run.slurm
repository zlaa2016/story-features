#!/usr/bin/env bash
#SBATCH --job-name=mech_norm
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

set -euo pipefail
source shared/slurm_env.sh

if [[ -z "${RETENTION_CSV:-}" ]]; then
  echo "Set RETENTION_CSV to original_detail_retention.csv" >&2
  exit 2
fi

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" analysis/representation/demixing/analyze.py \
  --retention-csv "$RETENTION_CSV" \
  --layer "${ANALYSIS_LAYER:-16}" \
  --components "${N_COMPONENTS:-32}"'
