#!/usr/bin/env bash
#SBATCH --job-name=jlens_fit
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/mechanistic_interpretability/jlens_fit-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_fit-%A_%a.err

set -euo pipefail
source hpc/slurm_env.shslurm_env.sh
JLENS_PYTHON="${JLENS_PYTHON:-/scratch/$USER/venvs/jlens/bin/python}"
SHARD="${SLURM_ARRAY_TASK_ID}"
START=$((SHARD * 10))
run_in_container "PYTHONPATH=src $JLENS_PYTHON src/analysis/representation/jacobian/jacobian_lens/fit.py --start $START --limit 10 --output results/analysis/representation/jacobian/jacobian_lens/shards/lens_$(printf '%02d' "$SHARD").pt"
