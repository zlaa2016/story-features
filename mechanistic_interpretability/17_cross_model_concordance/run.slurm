#!/usr/bin/env bash
#SBATCH --job-name=model_concordance
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.17_cross_model_concordance.analyze \
  --reference mechanistic_interpretability/outputs/13_structured_detail_alignment/story_alignment_metrics.csv \
  --open-model qwen3_8b mechanistic_interpretability/outputs/03_native_generation/qwen3_8b/alignment/story_alignment_metrics.csv \
  --open-model gpt_oss_20b mechanistic_interpretability/outputs/03_native_generation/gpt_oss_20b/alignment/story_alignment_metrics.csv \
  --output-dir mechanistic_interpretability/outputs/17_cross_model_concordance'
