#!/usr/bin/env bash
#SBATCH --job-name=mech_router
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" mechanistic_interpretability/07_router_analysis/analyze.py \
  --input-dir mechanistic_interpretability/outputs/02_representation_extraction/'"$MODEL_KEY"' \
  --output-dir mechanistic_interpretability/outputs/07_router_analysis/'"$MODEL_KEY"''
