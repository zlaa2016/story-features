#!/usr/bin/env bash
#SBATCH --job-name=enc_story_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
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

export MASTER="${MASTER:-data/processed/master_stories.parquet}"
export MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
export TEXT_COL="${TEXT_COL:-transcript_best}"
export ID_COL="${ID_COL:-video_id}"
export CLEANED_META="${CLEANED_META:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export CLEANED_COL="${CLEANED_COL:-cleaned}"
export MAX_LEN="${MAX_LEN:-128}"
export STRIDE="${STRIDE:-64}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export OUTDIR="${OUTDIR:-artifacts/steer/story_vectors/default}"

./env/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" features/representation_extraction/encoder/extract_story_vectors_encoder.py \
      --master "$MASTER" \
      --outdir "$OUTDIR" \
      --model "$MODEL_NAME" \
      --text-col "$TEXT_COL" \
      --id-col "$ID_COL" \
      --cleaned-meta "$CLEANED_META" \
      --cleaned-col "$CLEANED_COL" \
      --max-len "$MAX_LEN" \
      --stride "$STRIDE" \
      --batch-size "$BATCH_SIZE"'
