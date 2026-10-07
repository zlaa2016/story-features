#!/usr/bin/env bash
#SBATCH --job-name=mech_probes
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

RETENTION_CSV="${RETENTION_CSV:-mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/probes/layerwise_detail/run.py \
  --input-dir results/features/representation_extraction/causal_lm/detail_extraction/'"$MODEL_KEY"' \
  --output-dir results/analysis/representation/probes/layerwise_detail/'"$MODEL_KEY"' \
  --retention-csv "'"$RETENTION_CSV"'" \
  --folds "${FOLDS:-5}"'
