#!/usr/bin/env bash
#SBATCH --job-name=story_probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs artifacts/linear_probe

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export VECTORS="${VECTORS:-artifacts/distilbert_vectors/story_vectors.npz}"
export FEATURES_CSV="${FEATURES_CSV:-}"
export PROBE_OUTDIR="${PROBE_OUTDIR:-artifacts/linear_probe}"
export BINS="${BINS:-0-3,4-6,7-12}"
export SEED="${SEED:-42}"
export SHUFFLE_LABELS="${SHUFFLE_LABELS:-0}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   PROBE_ARGS=(--vectors "$VECTORS" --outdir "$PROBE_OUTDIR" --bins "$BINS" --seed "$SEED"); \
   if [[ -n "${FEATURES_CSV:-}" ]]; then PROBE_ARGS+=(--features-csv "$FEATURES_CSV"); fi; \
   if [[ "${SHUFFLE_LABELS:-0}" == "1" ]]; then PROBE_ARGS+=(--shuffle-labels); fi; \
   "$PYTHON_BIN" scripts/run_linear_probe.py "${PROBE_ARGS[@]}"'
