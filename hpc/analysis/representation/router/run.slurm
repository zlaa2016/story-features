#!/usr/bin/env bash
#SBATCH --job-name=mech_router
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/router/analyze.py \
  --input-dir results/features/representation_extraction/causal_lm/detail_extraction/'"$MODEL_KEY"' \
  --output-dir results/analysis/representation/router/'"$MODEL_KEY"''
