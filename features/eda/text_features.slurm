#!/usr/bin/env bash
#SBATCH --job-name=story_textfeat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs artifacts

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export BASE="${BASE:-data/amazon_data/all_extracted_titles_with_amazon.xlsx}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_stories.parquet}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_stories_index.csv}"
export FEATURES_OUT="${FEATURES_OUT:-artifacts/text_features.csv}"
export AOA_LEXICON="${AOA_LEXICON:-}"
export AOA_WORD_COL="${AOA_WORD_COL:-}"
export AOA_AGE_COL="${AOA_AGE_COL:-}"
export AOA_LOOKUP="${AOA_LOOKUP:-}"

./env/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" data_pipeline/cleaning/master_dataset/build_master_dataset.py --base "$BASE" --out "$MASTER_OUT" --out-index "$MASTER_INDEX" && \
   AOA_ARGS=(); \
   if [[ -n "${AOA_LEXICON:-}" ]]; then AOA_ARGS+=(--aoa-lexicon "$AOA_LEXICON"); fi; \
   if [[ -n "${AOA_WORD_COL:-}" ]]; then AOA_ARGS+=(--aoa-word-col "$AOA_WORD_COL"); fi; \
   if [[ -n "${AOA_AGE_COL:-}" ]]; then AOA_ARGS+=(--aoa-age-col "$AOA_AGE_COL"); fi; \
   if [[ -n "${AOA_LOOKUP:-}" ]]; then AOA_ARGS+=(--aoa-lookup "$AOA_LOOKUP"); fi; \
   "$PYTHON_BIN" features/eda/analyze_text_features.py --master "$MASTER_OUT" --out "$FEATURES_OUT" --extra-nlp --concepts-from-events "${AOA_ARGS[@]}"'
