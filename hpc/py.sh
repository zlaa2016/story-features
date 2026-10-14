#!/usr/bin/env bash
# Run a Python script inside the project container from a login node (the login node has no python3). Light jobs only; use sbatch for heavy work.
#     bash hpc/py.sh src/features/summaries/dataset_index/describe_cohort.py --vector-dir <dir>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY="${OVERLAY:-/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro}"
export CONDA_ENV="${CONDA_ENV:-story-features}"
export PYTHON_BIN="${PYTHON_BIN:-/ext3/${CONDA_ENV}/bin/python}"
export USE_NV=0
cmd='PYTHONPATH=src "$PYTHON_BIN"'
for arg in "$@"; do cmd+=" $(printf '%q' "$arg")"; done
exec ./hpc/run_in_singularity.sh "cd $(pwd) && $cmd"
