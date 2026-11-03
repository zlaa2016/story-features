#!/usr/bin/env bash
# Submit the deletion-graph extraction for one batch and one model.
#   MODEL_KEY=qwen3_8b BATCH=batch_top100_ages2-12 bash hpc/analysis/representation/geometry/deletion_graph/submit.sh
# Only some failed shards: ONLY_TASKS="3,7" with the same variables.  A shard whose output already exists is skipped.
set -euo pipefail
ACCOUNT="${ACCOUNT:-torch_pr_494_general}"
export MODEL_KEY="${MODEL_KEY:-qwen3_8b}" BATCH="${BATCH:-batch_top100_ages2-12}"
export STORIES_PER_SHARD="${STORIES_PER_SHARD:-20}" MAX_STORIES="${MAX_STORIES:-0}"
STORIES=$(( $(wc -l < "data/processed/spectral_age/$BATCH/subset_stories.csv") - 1 ))
[[ "$MAX_STORIES" -gt 0 && "$MAX_STORIES" -lt "$STORIES" ]] && STORIES="$MAX_STORIES"
SHARDS=$(( (STORIES + STORIES_PER_SHARD - 1) / STORIES_PER_SHARD ))
ARRAY="${ONLY_TASKS:-0-$((SHARDS - 1))}%4"
echo "$STORIES stories -> $SHARDS shards of $STORIES_PER_SHARD, model $MODEL_KEY"
sbatch --account="$ACCOUNT" --array="$ARRAY" --export=ALL hpc/analysis/representation/geometry/deletion_graph/extract.slurm
