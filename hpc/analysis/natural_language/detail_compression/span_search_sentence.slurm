#!/usr/bin/env bash
#SBATCH --job-name=span_nl_sent
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
source hpc/slurm_env.sh

# Sentence-level compression: how many full-text details does each summary SENTENCE cover? Same search, same stories,
# but the summary units come from the sentence-segmented file. No GPT-OSS vectors are involved.
SENTENCE_FILE="${SENTENCE_FILE:-results/features/summaries/runs/age_steering_pm3_customer_discrete_resume_gpt-5.4-mini_20260817_080237/detail_segments_current_subset_local_sentence/detail_segments_summaries.jsonl}"
export SENTENCE_FILE
test -s "$SENTENCE_FILE" || { echo "Missing $SENTENCE_FILE" >&2; exit 1; }

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/analysis/natural_language/detail_compression/span_search.py \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  --output-dir "${OUTPUT_DIR:-results/analysis/natural_language/detail_compression/span_search_sentence}" \
  --device "${DEVICE:-cuda}" \
  --model "${ALIGNMENT_MODEL:-sentence-transformers/all-mpnet-base-v2}" \
  --max-span "${MAX_SPAN:-10}" \
  --length-penalties "${LENGTH_PENALTIES:-0}" \
  --summary-segments "$SENTENCE_FILE" \
  --limit "${LIMIT:-0}"'
