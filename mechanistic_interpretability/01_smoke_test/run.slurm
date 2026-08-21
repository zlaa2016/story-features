#!/usr/bin/env bash
#SBATCH --job-name=mech_smoke
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" mechanistic_interpretability/01_smoke_test/run.py \
  --model-key "'"$MODEL_KEY"'" \
  --cohort mechanistic_interpretability/outputs/00_dataset_index/cohort.jsonl \
  --output mechanistic_interpretability/outputs/01_smoke_test/'"$MODEL_KEY"'/report.json'
