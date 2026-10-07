#!/usr/bin/env bash
#SBATCH --job-name=causal_geom_stats
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
BASE="results/analysis/representation/geometry/compression_geometry/${MODEL_KEY}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.natural_language.detail_alignment.merge_shards \
  --input-dir "'"$BASE"'/alignment" && \
PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.compression_geometry.analyze \
  --alignment-dir "'"$BASE"'/alignment" \
  --output-dir "'"$BASE"'/analysis"'
