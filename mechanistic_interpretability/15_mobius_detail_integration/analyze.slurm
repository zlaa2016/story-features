#!/usr/bin/env bash
#SBATCH --job-name=detail_mobius_stats
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.15_mobius_detail_integration.analyze \
  --input-dir mechanistic_interpretability/outputs/15_mobius_detail_integration'
