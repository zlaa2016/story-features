#!/usr/bin/env bash
#SBATCH --job-name=jlens_merge
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/mechanistic_interpretability/jlens_merge-%j.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_merge-%j.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
JLENS_PYTHON="${JLENS_PYTHON:-/scratch/$USER/venvs/jlens/bin/python}"
run_in_container "PYTHONPATH=. $JLENS_PYTHON mechanistic_interpretability/11_jacobian_lens/merge.py"
