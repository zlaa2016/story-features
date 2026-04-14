#!/usr/bin/env bash
#SBATCH --job-name=layer_probe_reg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs artifacts/layerwise_regression_probe

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"

# Required env vars:
#   VECTORS_NPZ : path to extracted vectors (e.g. artifacts/distilbert_vectors/story_vectors.npz)
#   FEATURES_CSV: path to features (e.g. artifacts/text_features_batch1.csv)
#   TARGET      : column name (e.g. flesch_kincaid_grade_est, aoa_mean)

./hpc/run_in_singularity.sh \
  "cd \"$PWD\" && \
   python scripts/run_layerwise_regression_probe.py --vectors \"${VECTORS_NPZ}\" --features \"${FEATURES_CSV}\" --target \"${TARGET}\" --outdir artifacts/layerwise_regression_probe"
