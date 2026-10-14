#!/usr/bin/env bash
#SBATCH --job-name=jlens_apply
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%4
#SBATCH --output=logs/mechanistic_interpretability/jlens_apply-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_apply-%A_%a.err

set -euo pipefail
source hpc/slurm_env.sh
JLENS_PYTHON="${JLENS_PYTHON:-/scratch/$USER/venvs/jlens/bin/python}"
SHARD="${SLURM_ARRAY_TASK_ID}"
START=$((SHARD * 500))
run_in_container "PYTHONPATH=src $JLENS_PYTHON src/analysis/representation/jacobian/jacobian_lens/apply.py --start $START --limit 500 --output results/analysis/representation/jacobian/jacobian_lens/readouts_$(printf '%02d' "$SHARD").jsonl"
