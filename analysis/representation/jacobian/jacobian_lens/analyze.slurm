#!/usr/bin/env bash
#SBATCH --job-name=jlens_analyze
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/mechanistic_interpretability/jlens_analyze-%j.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_analyze-%j.err

set -euo pipefail
source shared/slurm_env.sh
run_in_container "PYTHONPATH=. $PYTHON_BIN analysis/representation/jacobian/jacobian_lens/analyze.py"
