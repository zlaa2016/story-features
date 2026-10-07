#!/usr/bin/env bash
#SBATCH --job-name=layer_probe_reg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

export OUTDIR="${OUTDIR:-artifacts/layerwise_regression_probe}"
mkdir -p logs "$OUTDIR"

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"

# Required env vars:
#   VECTORS_NPZ : path to extracted vectors (e.g. results/features/representation_extraction/encoder/distilbert_vectors/story_vectors.npz)
#   FEATURES_CSV: path to features (e.g. results/features/eda/text_features_batch1.csv)
#   TARGET      : column name (e.g. flesch_kincaid_grade_est, aoa_mean)
#
# Back-compat aliases:
#   VECTORS  -> VECTORS_NPZ
#   FEATURES -> FEATURES_CSV

export VECTORS_NPZ="${VECTORS_NPZ:-${VECTORS:-}}"
export FEATURES_CSV="${FEATURES_CSV:-${FEATURES:-}}"

if [[ -z "${VECTORS_NPZ}" ]]; then
  echo "Missing required env var: VECTORS_NPZ (or alias VECTORS)" >&2
  exit 2
fi
if [[ -z "${FEATURES_CSV}" ]]; then
  echo "Missing required env var: FEATURES_CSV (or alias FEATURES)" >&2
  exit 2
fi
if [[ -z "${TARGET:-}" ]]; then
  echo "Missing required env var: TARGET" >&2
  exit 2
fi

./env/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" analysis/representation/probes/run_layerwise_regression_probe.py --vectors "$VECTORS_NPZ" --features "$FEATURES_CSV" --target "$TARGET" --outdir "$OUTDIR"'
