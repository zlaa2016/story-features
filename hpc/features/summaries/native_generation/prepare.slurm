#!/usr/bin/env bash
#SBATCH --job-name=native_prep
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m features.summaries.native_generation.prepare_alignment \
  --generation-dir results/features/summaries/native_generation/'"$MODEL_KEY"' \
  --output-dir results/features/summaries/native_generation/'"$MODEL_KEY"'/alignment_records'
