#!/usr/bin/env bash
#SBATCH --job-name=mech_steer
#SBATCH --array=0-19%4
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/steering/activation_steering/run.py \
  --model-key "'"$MODEL_KEY"'" \
  --directions "results/analysis/representation/probes/layerwise_detail/'"$MODEL_KEY"'/age_directions.npz" \
  --story-split "results/analysis/representation/probes/layerwise_detail/'"$MODEL_KEY"'/story_split.csv" \
  --direction-name "${DIRECTION_NAME:-older_minus_actual}" \
  --layer "${STEERING_LAYER:-16}" \
  --alphas "${ALPHAS:--2,-1,-0.5,0,0.5,1,2}" \
  --shard-index "'"$SLURM_ARRAY_TASK_ID"'" --num-shards 20 \
  --output "results/analysis/representation/steering/activation_steering/'"$MODEL_KEY"'/generations_'"$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"'.jsonl"'
