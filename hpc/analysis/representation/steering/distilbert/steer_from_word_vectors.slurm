#!/usr/bin/env bash
#SBATCH --job-name=story_steer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs results/analysis/representation/steering/distilbert/steer

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"

export STORY_VECTORS="${STORY_VECTORS:-artifacts/distilbert_vectors/story_vectors.npz}"
export WORD_VECTORS="${WORD_VECTORS:-artifacts/steer/word_vectors/default/word_vectors.npz}"
export FEATURES="${FEATURES:-artifacts/text_features_batch1.csv}"
export STORY_AOA_STATS="${STORY_AOA_STATS:-artifacts/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv}"
export CLEANED_META="${CLEANED_META:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export CLEANED_COL="${CLEANED_COL:-cleaned}"
export AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
export STORY_AOA_COL="${STORY_AOA_COL:-aoa_mean_story}"
export STORY_AOA_LOW_Q="${STORY_AOA_LOW_Q:-0.33}"
export STORY_AOA_HIGH_Q="${STORY_AOA_HIGH_Q:-0.67}"
export CONTROL="${CONTROL:-real}"
export CONTROL_SEED="${CONTROL_SEED:-42}"
export SEED="${SEED:-42}"
export OUTDIR="${OUTDIR:-artifacts/steer/results/default}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" src/analysis/representation/steering/distilbert/run_story_steering_from_word_vectors.py \
      --story-vectors "$STORY_VECTORS" \
      --word-vectors "$WORD_VECTORS" \
      --features "$FEATURES" \
      --story-aoa-stats "$STORY_AOA_STATS" \
      --cleaned-meta "$CLEANED_META" \
      --cleaned-col "$CLEANED_COL" \
      --age-cutoff "$AGE_CUTOFF" \
      --story-aoa-col "$STORY_AOA_COL" \
      --story-aoa-low-quantile "$STORY_AOA_LOW_Q" \
      --story-aoa-high-quantile "$STORY_AOA_HIGH_Q" \
      --control "$CONTROL" \
      --control-seed "$CONTROL_SEED" \
      --seed "$SEED" \
      --outdir "$OUTDIR"'
