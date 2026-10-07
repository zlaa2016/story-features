#!/usr/bin/env bash
#SBATCH --job-name=detail_mobius_stats
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.sh
run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.natural_language.mobius_integration.analyze \
  --input-dir results/analysis/natural_language/mobius_integration'
