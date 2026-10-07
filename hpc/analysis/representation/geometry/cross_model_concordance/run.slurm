#!/usr/bin/env bash
#SBATCH --job-name=model_concordance
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.sh
run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.cross_model_concordance.analyze \
  --reference results/analysis/natural_language/detail_alignment/story_alignment_metrics.csv \
  --open-model qwen3_8b results/features/summaries/native_generation/qwen3_8b/alignment/story_alignment_metrics.csv \
  --open-model gpt_oss_20b results/features/summaries/native_generation/gpt_oss_20b/alignment/story_alignment_metrics.csv \
  --output-dir results/analysis/representation/geometry/cross_model_concordance'
