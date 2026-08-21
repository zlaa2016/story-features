#!/usr/bin/env bash
#SBATCH --job-name=jlens_analyze
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mechanistic_interpretability/jlens_analyze-%j.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_analyze-%j.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
run_in_container "PYTHONPATH=. $PYTHON_BIN mechanistic_interpretability/11_jacobian_lens/analyze.py"
