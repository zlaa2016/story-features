#!/usr/bin/env bash
#SBATCH --job-name=dec_rewrite
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

mkdir -p logs results/analysis/representation/steering/decoder_rewrite/decoder_rewrites

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

export MASTER="${MASTER:-data/processed/master_stories.parquet}"
export STORY_VECTORS="${STORY_VECTORS:-}"
export WORD_VECTORS="${WORD_VECTORS:-}"
export CLEANED_META="${CLEANED_META:-story_wAmazonData_042626/amazon_youtube_batch1_wid.csv}"
export CLEANED_COL="${CLEANED_COL:-cleaned}"
export STORY_AOA_STATS="${STORY_AOA_STATS:-artifacts/corpus_aoa_cutoffs_cleaned_idchecked/story_aoa_stats.csv}"
export TEXT_COL="${TEXT_COL:-transcript_best}"
export ID_COL="${ID_COL:-video_id}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-Base}"
export AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
export DIRECTION_TRAIN_FRAC="${DIRECTION_TRAIN_FRAC:-0.5}"
export EVALUATOR_VAL_SIZE="${EVALUATOR_VAL_SIZE:-0.15}"
export REWRITE_TEST_SIZE="${REWRITE_TEST_SIZE:-0.15}"
export LAYER="${LAYER:-0}"
export ALPHA="${ALPHA:-1.0}"
export MAX_ITEMS="${MAX_ITEMS:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export REASONING="${REASONING:-low}"
export OUTDIR="${OUTDIR:-artifacts/decoder_rewrites/default}"
export REWRITE_PROMPT="${REWRITE_PROMPT:-Rewrite this story for slightly older children using somewhat more advanced vocabulary and sentence structure, while preserving the same plot, characters, and meaning.}"

./env/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   "$PYTHON_BIN" analysis/representation/steering/decoder_rewrite/activation_steer_decoder_rewrite.py \
      --master "$MASTER" \
      --story-vectors "$STORY_VECTORS" \
      --word-vectors "$WORD_VECTORS" \
      --cleaned-meta "$CLEANED_META" \
      --cleaned-col "$CLEANED_COL" \
      --story-aoa-stats "$STORY_AOA_STATS" \
      --text-col "$TEXT_COL" \
      --id-col "$ID_COL" \
      --model "$MODEL_NAME" \
      --age-cutoff "$AGE_CUTOFF" \
      --direction-train-frac "$DIRECTION_TRAIN_FRAC" \
      --evaluator-val-size "$EVALUATOR_VAL_SIZE" \
      --rewrite-test-size "$REWRITE_TEST_SIZE" \
      --layer "$LAYER" \
      --alpha "$ALPHA" \
      --max-items "$MAX_ITEMS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --reasoning "$REASONING" \
      --outdir "$OUTDIR" \
      --rewrite-prompt "$REWRITE_PROMPT"'
