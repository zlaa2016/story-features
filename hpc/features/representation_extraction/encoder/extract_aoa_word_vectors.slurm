#!/usr/bin/env bash
#SBATCH --job-name=aoa_word_vecs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

mkdir -p logs results/analysis/representation/steering/distilbert/steer

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export HF_TOKEN_DIR="${HF_TOKEN_DIR:-/scratch/$USER/.cache/huggingface}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_DIR}/token" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_DIR}/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

export MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
export MODEL_FAMILY="${MODEL_FAMILY:-encoder}"
export AOA_LEXICON="${AOA_LEXICON:-data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx}"
export WORD_AOA_LOW="${WORD_AOA_LOW:-4.0}"
export WORD_AOA_HIGH="${WORD_AOA_HIGH:-9.0}"
export REASONING="${REASONING:-low}"
export MAX_LENGTH="${MAX_LENGTH:-64}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export OUTDIR="${OUTDIR:-results/analysis/representation/steering/distilbert/steer/word_vectors/default}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" src/features/representation_extraction/encoder/extract_aoa_word_vectors.py \
      --aoa-lexicon "$AOA_LEXICON" \
      --model "$MODEL_NAME" \
      --family "$MODEL_FAMILY" \
      --outdir "$OUTDIR" \
      --low "$WORD_AOA_LOW" \
      --high "$WORD_AOA_HIGH" \
      --reasoning "$REASONING" \
      --max-length "$MAX_LENGTH" \
      --batch-size "$BATCH_SIZE"'
