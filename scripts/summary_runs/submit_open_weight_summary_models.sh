#!/usr/bin/env bash
set -euo pipefail

# Submit one local open-weight summary job per model key.
# Override MODEL_KEYS to restrict the set, e.g.:
#   MODEL_KEYS="gpt_oss_20b qwen3_8b olmo_2_7b_instruct" bash scripts/summary_runs/submit_open_weight_summary_models.sh

MODEL_KEYS="${MODEL_KEYS:-gpt_oss_20b qwen3_8b qwen3_14b qwen3_30b_a3b_instruct_2507 gemma_3_1b_it olmo_2_7b_instruct}"

for key in $MODEL_KEYS; do
  echo "Submitting open-weight summary job for ${key}"
  sbatch --export=ALL,OPEN_WEIGHT_SUMMARY_MODEL_KEY="${key}" scripts/summary_runs/run_open_weight_summaries.slurm
done
