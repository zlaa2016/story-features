#!/usr/bin/env bash
#SBATCH --job-name=story_textfeat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs artifacts

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-stroy}"
export USE_NV="${USE_NV:-0}"

./hpc/run_in_singularity.sh \
  "cd \"$PWD\" && \
   MASTER=data/processed/master_stories.parquet; \
   if [[ ! -f \"${MASTER}\" && -f data/processed/master_stories.csv.gz ]]; then MASTER=data/processed/master_stories.csv.gz; fi; \
   python scripts/analyze_text_features.py --master \"${MASTER}\" --out artifacts/text_features.csv"

