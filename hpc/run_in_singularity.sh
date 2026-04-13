#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run a command inside your Singularity image + overlay.
#
# Required env vars:
#   SIF      : e.g. /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
#   OVERLAY  : e.g. /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro
#
# Optional env vars:
#   USE_NV=1 : pass --nv (default: 1)
#   FAKEROOT=1 : pass --fakeroot (default: 1)
#   CONDA_SH : path to conda.sh inside container (default: /opt/conda/etc/profile.d/conda.sh)
#   CONDA_ENV: conda env name to activate (default: story)

USE_NV="${USE_NV:-1}"
FAKEROOT="${FAKEROOT:-1}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-story}"

if [[ -z "${SIF:-}" ]]; then
  echo "Missing SIF env var" >&2
  exit 2
fi
if [[ -z "${OVERLAY:-}" ]]; then
  echo "Missing OVERLAY env var" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: SIF=... OVERLAY=... $0 <command...>" >&2
  exit 2
fi

nv_args=()
fakeroot_args=()
if [[ "$USE_NV" == "1" ]]; then nv_args+=(--nv); fi
if [[ "$FAKEROOT" == "1" ]]; then fakeroot_args+=(--fakeroot); fi

cmd="$*"

singularity exec \
  "${fakeroot_args[@]}" \
  "${nv_args[@]}" \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  /bin/bash -lc "set -euo pipefail; if [[ -f \"${CONDA_SH}\" ]]; then source \"${CONDA_SH}\"; conda activate \"${CONDA_ENV}\"; fi; ${cmd}"

