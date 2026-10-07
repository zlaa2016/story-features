#!/usr/bin/env bash
#SBATCH --job-name=eval_rewrites
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs artifacts

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export USE_NV="${USE_NV:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export HF_TOKEN_DIR="${HF_TOKEN_DIR:-/scratch/$USER/.cache/huggingface}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_DIR}/token" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_DIR}/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

export INPUT_CSV="${INPUT_CSV:-}"
export ORIGINAL_COL="${ORIGINAL_COL:-original_text}"
export REWRITTEN_COL="${REWRITTEN_COL:-rewritten_text}"
export AOA_LEXICON="${AOA_LEXICON:-data/aoa/AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx}"
export AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
export AGE_MODEL_DIR="${AGE_MODEL_DIR:-}"
export PPL_MODEL="${PPL_MODEL:-}"
export PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-2}"
export PPL_MAX_LENGTH="${PPL_MAX_LENGTH:-1024}"
export OUTDIR="${OUTDIR:-results/analysis/natural_language/rewrite_evaluation/rewrite_eval/default}"

./hpc/run_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && \
   EVAL_ARGS=(--input-csv "$INPUT_CSV" --original-col "$ORIGINAL_COL" --rewritten-col "$REWRITTEN_COL" --aoa-lexicon "$AOA_LEXICON" --age-cutoff "$AGE_CUTOFF" --outdir "$OUTDIR"); \
   if [[ -n "${AGE_MODEL_DIR:-}" ]]; then EVAL_ARGS+=(--age-model-dir "$AGE_MODEL_DIR"); fi; \
   if [[ -n "${PPL_MODEL:-}" ]]; then EVAL_ARGS+=(--ppl-model "$PPL_MODEL" --ppl-batch-size "$PPL_BATCH_SIZE" --ppl-max-length "$PPL_MAX_LENGTH"); fi; \
   "$PYTHON_BIN" src/analysis/natural_language/rewrite_evaluation/evaluate_rewrites.py "${EVAL_ARGS[@]}"'
