#!/usr/bin/env bash
#SBATCH --job-name=story_ppl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

mkdir -p logs artifacts/perplexity/runs data/processed

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export USE_NV="${USE_NV:-1}"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_TOKEN_DIR="${HF_TOKEN_DIR:-/scratch/$USER/.cache/huggingface}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_DIR}/token" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_DIR}/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

export BASE="${BASE:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-story_wAmazonData_042626/cleaned_idchecked}"
export TRANSCRIPTS_ZIP="${TRANSCRIPTS_ZIP:-}"
export TRANSCRIPTS_SOURCE_NAME="${TRANSCRIPTS_SOURCE_NAME:-}"
export MASTER_OUT="${MASTER_OUT:-data/processed/master_stories.parquet}"
export MASTER_CSV="${MASTER_CSV:-data/processed/master_stories.csv.gz}"
export MASTER_INDEX="${MASTER_INDEX:-data/processed/master_stories_index.csv}"
export PPL_DATASET="${PPL_DATASET:-artifacts/perplexity/stories_for_perplexity.parquet}"
export PPL_OUTDIR="${PPL_OUTDIR:-artifacts/perplexity/runs}"
export MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
export MAX_LENGTH="${MAX_LENGTH:-0}"
export STRIDE="${STRIDE:-1024}"
export OVERLENGTH_POLICY="${OVERLENGTH_POLICY:-skip}"
export LIMIT="${LIMIT:-0}"
export CLEANED_ONLY="${CLEANED_ONLY:-0}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   BUILD_ARGS=(--base "$BASE" --transcripts-dir "$TRANSCRIPTS_DIR" --out "$MASTER_OUT" --out-csv "$MASTER_CSV" --out-index "$MASTER_INDEX"); \
   if [[ -n "$TRANSCRIPTS_ZIP" ]]; then BUILD_ARGS+=(--transcripts-zip "$TRANSCRIPTS_ZIP"); fi; \
   if [[ -n "$TRANSCRIPTS_SOURCE_NAME" ]]; then BUILD_ARGS+=(--transcripts-source-name "$TRANSCRIPTS_SOURCE_NAME"); fi; \
   "$PYTHON_BIN" scripts/build_master_dataset.py "${BUILD_ARGS[@]}" && \
   PREP_ARGS=(); \
   if [[ "$CLEANED_ONLY" == "1" ]]; then PREP_ARGS+=(--cleaned-only); fi; \
   "$PYTHON_BIN" perplexity/prepare_dataset.py --master "$MASTER_OUT" --out "$PPL_DATASET" "${PREP_ARGS[@]}" --dedupe && \
   "$PYTHON_BIN" perplexity/run_perplexity.py \
      --dataset "$PPL_DATASET" \
      --output "$PPL_OUTDIR/${MODEL_KEY}.csv" \
      --model-key "$MODEL_KEY" \
      --max-length "$MAX_LENGTH" \
      --stride "$STRIDE" \
      --overlength-policy "$OVERLENGTH_POLICY" \
      --limit "$LIMIT" \
      --resume'
