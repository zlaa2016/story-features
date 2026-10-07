#!/usr/bin/env bash
#SBATCH --job-name=story_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs results/features/representation_extraction/encoder/distilbert_vectors data/processed

# Edit these for your cluster (or pass via env when submitting).
export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export BASE="${BASE:-data/amazon_data/all_extracted_titles_with_amazon.xlsx}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_stories.parquet}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_stories_index.csv}"
export VEC_OUTDIR="${VEC_OUTDIR:-artifacts/distilbert_vectors}"
export VEC_UNIT="${VEC_UNIT:-story}"
export VEC_SEGMENT_AGG="${VEC_SEGMENT_AGG:-mean}"

# If conda isn't at /opt/conda in your image, set CONDA_SH too.
# export CONDA_SH="/path/inside/container/to/conda.sh"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   mkdir -p "$VEC_OUTDIR" && \
   "$PYTHON_BIN" src/data_pipeline/cleaning/master_dataset/build_master_dataset.py --base "$BASE" --out "$MASTER_OUT" --out-index "$MASTER_INDEX" && \
   "$PYTHON_BIN" src/features/representation_extraction/encoder/extract_vectors_distilbert.py --master "$MASTER_OUT" --outdir "$VEC_OUTDIR" --unit "$VEC_UNIT" --segment-agg "$VEC_SEGMENT_AGG" && \
   test -f "$VEC_OUTDIR/story_vectors.npz" && ls -lh "$VEC_OUTDIR/story_vectors.npz"'
