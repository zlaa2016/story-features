#!/usr/bin/env bash
#SBATCH --job-name=mech_routepatch
#SBATCH --array=0-19%4
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

set -euo pipefail
source hpc/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/representation/router/causal_patch.py \
  --model-key "'"$MODEL_KEY"'" \
  --layers "${ROUTER_PATCH_LAYERS:-8,12,16,20}" \
  --story-split "results/analysis/representation/probes/layerwise_detail/'"$MODEL_KEY"'/story_split.csv" \
  --shard-index "'"$SLURM_ARRAY_TASK_ID"'" --num-shards 20 \
  --output "results/analysis/representation/router/'"$MODEL_KEY"'/causal_router_patch_'"$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"'.jsonl"'
