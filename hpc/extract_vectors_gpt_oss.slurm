#!/usr/bin/env bash
#SBATCH --job-name=gpt_oss_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

mkdir -p logs artifacts/gpt_oss_vectors data/processed

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-stroy}"

# Optional: point Hugging Face cache to scratch
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

./hpc/run_in_singularity.sh \
  "cd \"$PWD\" && \
   python scripts/build_master_dataset.py --out data/processed/master_stories.parquet --out-index data/processed/master_stories_index.csv && \
   MASTER=data/processed/master_stories.parquet; \
   if [[ ! -f \"${MASTER}\" && -f data/processed/master_stories.csv.gz ]]; then MASTER=data/processed/master_stories.csv.gz; fi; \
   python scripts/extract_vectors_gpt_oss.py --master \"${MASTER}\" --outdir artifacts/gpt_oss_vectors --model openai/gpt-oss-20b --unit story --reasoning low"

