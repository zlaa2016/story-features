#!/usr/bin/env bash
#SBATCH --job-name=native_prep
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.03_native_generation.prepare_alignment \
  --generation-dir mechanistic_interpretability/outputs/03_native_generation/'"$MODEL_KEY"' \
  --output-dir mechanistic_interpretability/outputs/03_native_generation/'"$MODEL_KEY"'/alignment_records'
