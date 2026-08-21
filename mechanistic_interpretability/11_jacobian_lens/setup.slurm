#!/usr/bin/env bash
#SBATCH --job-name=jlens_setup
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/mechanistic_interpretability/jlens_setup-%j.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_setup-%j.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

JLENS_REPO="${JLENS_REPO:-/scratch/$USER/tools/jacobian-lens}"
JLENS_VENV="${JLENS_VENV:-/scratch/$USER/venvs/jlens}"
run_in_container "mkdir -p \"$(dirname "$JLENS_REPO")\" \"$(dirname "$JLENS_VENV")\" && if [[ -d \"$JLENS_REPO/.git\" ]]; then git -C \"$JLENS_REPO\" pull --ff-only; else git clone --depth 1 https://github.com/anthropics/jacobian-lens.git \"$JLENS_REPO\"; fi && $PYTHON_BIN -m venv --system-site-packages \"$JLENS_VENV\" && \"$JLENS_VENV/bin/python\" -m pip install --upgrade pip && \"$JLENS_VENV/bin/python\" -m pip install -e \"$JLENS_REPO\""
