#!/usr/bin/env bash
#SBATCH --job-name=feat_base_cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs artifacts data/processed

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"

# Inputs/outputs (override via env on sbatch if desired)
export BASE="${BASE:-data/amazon_data/amazon_youtube_batch1.csv}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_batch1.parquet}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_batch1_index.csv}"
export FEATURES_OUT="${FEATURES_OUT:-artifacts/text_features_batch1_review_samples.csv}"
export BASELINE_OUT="${BASELINE_OUT:-artifacts/feature_baseline_batch1_review_samples.csv}"
export BINS="${BINS:-0-3,4-6,7-12}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" scripts/build_master_dataset.py --base "$BASE" --out "$MASTER_OUT" --out-index "$MASTER_INDEX" && \
   "$PYTHON_BIN" scripts/analyze_text_features.py --master "$MASTER_OUT" --out "$FEATURES_OUT" --text-from review_events --extra-nlp --concepts-from-events && \
   "$PYTHON_BIN" scripts/run_feature_baseline_probe.py --features "$FEATURES_OUT" --bins "$BINS" --out "$BASELINE_OUT"'
