#!/usr/bin/env bash
#SBATCH --job-name=age_ft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

mkdir -p logs artifacts

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"

export MASTER="${MASTER:-data/processed/master_stories.parquet}"
export CLEANED_META="${CLEANED_META:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export CLEANED_COL="${CLEANED_COL:-cleaned}"
export TEXT_COL="${TEXT_COL:-transcript_best}"
export MODEL_NAME="${MODEL_NAME:-roberta-base}"
export AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
export OUTDIR="${OUTDIR:-artifacts/finetuned_age_classifier/roberta_base}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" scripts/finetune_age_text_classifier.py \
      --master "$MASTER" \
      --cleaned-meta "$CLEANED_META" \
      --cleaned-col "$CLEANED_COL" \
      --text-col "$TEXT_COL" \
      --model "$MODEL_NAME" \
      --age-cutoff "$AGE_CUTOFF" \
      --outdir "$OUTDIR"'
