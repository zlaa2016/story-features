#!/usr/bin/env bash
#SBATCH --job-name=jlens_swap
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%4
#SBATCH --output=logs/mechanistic_interpretability/jlens_swap-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/jlens_swap-%A_%a.err

set -euo pipefail
source shared/slurm_env.sh
JLENS_PYTHON="${JLENS_PYTHON:-/scratch/$USER/venvs/jlens/bin/python}"
run_in_container "PYTHONPATH=. $JLENS_PYTHON analysis/representation/jacobian/jacobian_lens/intervene.py --shard-index $SLURM_ARRAY_TASK_ID --num-shards 20 --output results/analysis/representation/jacobian/jacobian_lens/coordinate_swaps_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
