#!/usr/bin/env bash
#SBATCH --job-name=story_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs artifacts/distilbert_vectors data/processed

# Edit these for your cluster (or pass via env when submitting).
export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-stroy}"

# If conda isn't at /opt/conda in your image, set CONDA_SH too.
# export CONDA_SH="/path/inside/container/to/conda.sh"

./hpc/run_in_singularity.sh \
  "cd \"$PWD\" && \
   python scripts/build_master_dataset.py --out data/processed/master_stories.parquet --out-index data/processed/master_stories_index.csv && \
   MASTER=data/processed/master_stories.parquet; \
   if [[ ! -f \"${MASTER}\" && -f data/processed/master_stories.csv.gz ]]; then MASTER=data/processed/master_stories.csv.gz; fi; \
   python scripts/extract_vectors_distilbert.py --master \"${MASTER}\" --outdir artifacts/distilbert_vectors"
