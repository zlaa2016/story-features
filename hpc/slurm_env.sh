#!/usr/bin/env bash

set -euo pipefail

export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_TOKEN_DIR="${HF_TOKEN_DIR:-/scratch/$USER/.cache/huggingface}"
export USE_NV="${USE_NV:-1}"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_DIR}/token" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_DIR}/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

mkdir -p logs/mechanistic_interpretability

run_in_container() {
  local command="$1"
  if [[ -x hpc/run_in_singularity.sh ]]; then
    ./hpc/run_in_singularity.sh "cd \"$SLURM_SUBMIT_DIR\" && $command"
  else
    singularity exec --nv --fakeroot --overlay "$OVERLAY" "$SIF" /bin/bash -lc \
      "cd \"$SLURM_SUBMIT_DIR\" && $command"
  fi
}
