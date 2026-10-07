#!/usr/bin/env bash
#SBATCH --job-name=steer_aoa_words
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

mkdir -p logs results/analysis/representation/steering/distilbert/steer

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"

export VECTORS="${VECTORS:-artifacts/distilbert_vectors/story_vectors.npz}"
export FEATURES="${FEATURES:-artifacts/text_features_batch1.csv}"
export CLEANED_META="${CLEANED_META:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export CLEANED_COL="${CLEANED_COL:-cleaned}"
export STORY_AOA_STATS="${STORY_AOA_STATS:-artifacts/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv}"
export AOA_LEXICON="${AOA_LEXICON:-data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx}"
export MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
export AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
export STORY_AOA_COL="${STORY_AOA_COL:-aoa_mean_story}"
export STORY_AOA_LOW_Q="${STORY_AOA_LOW_Q:-0.33}"
export STORY_AOA_HIGH_Q="${STORY_AOA_HIGH_Q:-0.67}"
export WORD_AOA_LOW="${WORD_AOA_LOW:-4.0}"
export WORD_AOA_HIGH="${WORD_AOA_HIGH:-9.0}"
export SEED="${SEED:-42}"
export OUTDIR="${OUTDIR:-artifacts/steer/fixed_aoa_words}"

./env/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" analysis/representation/steering/distilbert/run_fixed_aoa_word_steering.py \
      --vectors "$VECTORS" \
      --features "$FEATURES" \
      --cleaned-meta "$CLEANED_META" \
      --cleaned-col "$CLEANED_COL" \
      --story-aoa-stats "$STORY_AOA_STATS" \
      --aoa-lexicon "$AOA_LEXICON" \
      --model "$MODEL_NAME" \
      --age-cutoff "$AGE_CUTOFF" \
      --story-aoa-col "$STORY_AOA_COL" \
      --story-aoa-low-quantile "$STORY_AOA_LOW_Q" \
      --story-aoa-high-quantile "$STORY_AOA_HIGH_Q" \
      --word-aoa-low "$WORD_AOA_LOW" \
      --word-aoa-high "$WORD_AOA_HIGH" \
      --seed "$SEED" \
      --outdir "$OUTDIR"'
