#!/usr/bin/env bash
#SBATCH --job-name=gpt_oss_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs results/features/representation_extraction/causal_lm/gpt_oss_vectors data/processed

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export HF_TOKEN_DIR="${HF_TOKEN_DIR:-/scratch/$USER/.cache/huggingface}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_DIR}/token" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_DIR}/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi
export BASE="${BASE:-data/amazon_data/all_extracted_titles_with_amazon.xlsx}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_stories.parquet}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_stories_index.csv}"
export VEC_OUTDIR="${VEC_OUTDIR:-artifacts/gpt_oss_vectors}"
export GPT_OSS_MODEL="${GPT_OSS_MODEL:-openai/gpt-oss-20b}"
export GPT_OSS_UNIT="${GPT_OSS_UNIT:-story}"
export GPT_OSS_REASONING="${GPT_OSS_REASONING:-low}"

# Optional: point Hugging Face cache to scratch
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   mkdir -p "$VEC_OUTDIR" && \
   "$PYTHON_BIN" src/data_pipeline/cleaning/master_dataset/build_master_dataset.py --base "$BASE" --out "$MASTER_OUT" --out-index "$MASTER_INDEX" && \
   "$PYTHON_BIN" src/features/representation_extraction/causal_lm/extract_vectors_gpt_oss.py --master "$MASTER_OUT" --outdir "$VEC_OUTDIR" --model "$GPT_OSS_MODEL" --unit "$GPT_OSS_UNIT" --reasoning "$GPT_OSS_REASONING" && \
   test -f "$VEC_OUTDIR/story_vectors.npz" && ls -lh "$VEC_OUTDIR/story_vectors.npz"'
