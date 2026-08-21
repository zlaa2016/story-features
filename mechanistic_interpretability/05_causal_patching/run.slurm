#!/usr/bin/env bash
#SBATCH --job-name=mech_patch
#SBATCH --array=0-19%4
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
run_in_container 'PYTHONPATH=. "$PYTHON_BIN" mechanistic_interpretability/05_causal_patching/run.py \
  --model-key "'"$MODEL_KEY"'" \
  --layers "${PATCH_LAYERS:-8,12,16,20}" \
  --story-split "mechanistic_interpretability/outputs/04_layerwise_probes/'"$MODEL_KEY"'/story_split.csv" \
  --shard-index "'"$SLURM_ARRAY_TASK_ID"'" --num-shards 20 \
  --output "mechanistic_interpretability/outputs/05_causal_patching/'"$MODEL_KEY"'/results_'"$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"'.jsonl"'
