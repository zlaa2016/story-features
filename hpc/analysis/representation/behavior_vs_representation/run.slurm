#!/usr/bin/env bash
#SBATCH --job-name=mech_dissoc
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/behavior_vs_representation/analyze.py \
  --input-dir results/features/representation_extraction/causal_lm/detail_extraction/'"$MODEL_KEY"' \
  --output-dir results/analysis/representation/behavior_vs_representation/'"$MODEL_KEY"''
