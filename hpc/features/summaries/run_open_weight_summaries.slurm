#!/usr/bin/env bash
#SBATCH --job-name=open_story_summ
#SBATCH --output=logs/summary_prompt_runs/open_weight_%j.out
#SBATCH --error=logs/summary_prompt_runs/open_weight_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

mkdir -p logs/summary_prompt_runs results/features/summaries/runs

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

export OPEN_WEIGHT_SUMMARY_MODEL_KEY="${OPEN_WEIGHT_SUMMARY_MODEL_KEY:-gpt_oss_20b}"
export SUMMARY_LIMIT="${SUMMARY_LIMIT:-10}"
export SUMMARY_START="${SUMMARY_START:-0}"
export SUMMARY_MAX_NEW_TOKENS="${SUMMARY_MAX_NEW_TOKENS:-1400}"
export SUMMARY_TEMPERATURE="${SUMMARY_TEMPERATURE:-0.2}"
export SUMMARY_OVERLENGTH_POLICY="${SUMMARY_OVERLENGTH_POLICY:-skip}"

if [[ -z "${SUMMARY_ARGS:-}" ]]; then
  SUMMARY_ARGS="--start $SUMMARY_START --limit $SUMMARY_LIMIT --model-key $OPEN_WEIGHT_SUMMARY_MODEL_KEY --max-new-tokens $SUMMARY_MAX_NEW_TOKENS --temperature $SUMMARY_TEMPERATURE --overlength-policy $SUMMARY_OVERLENGTH_POLICY"
fi

echo "Repo: ${SLURM_SUBMIT_DIR}"
echo "Model key: ${OPEN_WEIGHT_SUMMARY_MODEL_KEY}"
echo "Args: ${SUMMARY_ARGS}"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"

./hpc/run_in_singularity.shrun_in_singularity.sh \
  'cd "$SLURM_SUBMIT_DIR" && "$PYTHON_BIN" src/features/summaries/generate_open_weight_summaries.py $SUMMARY_ARGS'

echo "Finished: $(date)"

