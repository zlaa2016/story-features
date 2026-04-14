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
export BASE="${BASE:-data/amazon_data/all_extracted_titles_with_amazon.xlsx}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_stories.parquet}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_stories_index.csv}"
export FEATURES_OUT="${FEATURES_OUT:-artifacts/text_features.csv}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   python scripts/build_master_dataset.py --base "$BASE" --out "$MASTER_OUT" --out-index "$MASTER_INDEX" && \
   python scripts/analyze_text_features.py --master "$MASTER_OUT" --out "$FEATURES_OUT" --extra-nlp --concepts-from-events'
