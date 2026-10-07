#!/usr/bin/env bash
#SBATCH --job-name=causal_geom_prep
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.compression_geometry.prepare \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --generation-dir results/analysis/representation/steering/activation_steering/'"$MODEL_KEY"' \
  --output-dir results/analysis/representation/geometry/compression_geometry/'"$MODEL_KEY"'/records'
