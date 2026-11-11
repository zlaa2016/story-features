#!/usr/bin/env bash
#SBATCH --job-name=deletion_graph
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G

# Deletion-coupling matrices of the batch's stories (one forward pass per deleted detail), one array task = one shard of stories.
# Submit through submit.sh, which sets the array size.  Variables: MODEL_KEY (qwen3_8b|gpt_oss_20b), BATCH (folder under
# data/processed/spectral_age/), UNITS (details|events), STORIES_PER_SHARD, MAX_STORIES (0 = all), ATTN_IMPLEMENTATION (sdpa avoids the n^2 attention matrix).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
BATCH="${BATCH:-batch_top100_ages2-12}"
PER="${STORIES_PER_SHARD:-20}"
ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
# UNITS=events segments by the legacy event files (data/legacy/review_samples_event_segments) instead of the details
UNITS="${UNITS:-details}"
TAG=""; UNIT_ARGS=""
if [[ "$UNITS" == "events" ]]; then
  TAG="events_"
fi
OUT="${OUTPUT_DIR:-$ROOT/${MODEL_KEY}_deletion_graph_${TAG}${BATCH}}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD="$(printf "%03d" "$TASK")"
# the batch stories' original records live in the first cohort that has them: the main cohort, then the spectral-age cohorts
COHORTS="results/features/summaries/dataset_index/cohort.jsonl"
for FILE in "$ROOT"/spectral_age_cohort*.jsonl; do
  [[ -f "$FILE" ]] && COHORTS="$COHORTS $FILE"
done

if [[ "$UNITS" == "events" ]]; then
  UNIT_ARGS="--event-dir data/legacy/review_samples_event_segments"
else
  UNIT_ARGS="--cohort $COHORTS"
fi

run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" src/features/representation_extraction/causal_lm/detail_extraction/extract_deletion_graph.py \
  $UNIT_ARGS --stories data/processed/spectral_age/$BATCH/subset_stories.csv --max-stories ${MAX_STORIES:-0} \
  --model-key $MODEL_KEY --attn-implementation ${ATTN_IMPLEMENTATION:-sdpa} --layers ${LAYERS:-0,8,16,-1} \
  --max-tokens ${MAX_TOKENS:-32768} --start $((TASK * PER)) --limit $PER --shard-id $SHARD --output-dir $OUT"
