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
#   ENV_SH : path to an env bootstrap script inside container, e.g. /ext3/env.sh
#   CONDA_SH : path to conda.sh inside container (auto-detected if unset)
#   CONDA_ENV: conda env name to activate (default: story)
#   PYTHON_BIN: explicit python path inside container; adds its dir to PATH

USE_NV="${USE_NV:-1}"
FAKEROOT="${FAKEROOT:-1}"
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

activation_snippet='
set -euo pipefail

if [[ -n "${ENV_SH:-}" && -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}" || true
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PYTHON_BIN}" ]]; then
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  fi
fi

conda_sh=""
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  conda_sh="${CONDA_SH}"
else
  for c in \
    /opt/conda/etc/profile.d/conda.sh \
    /opt/conda3/etc/profile.d/conda.sh \
    /ext3/miniconda3/etc/profile.d/conda.sh \
    /ext3/miniconda/etc/profile.d/conda.sh \
    /ext3/conda/etc/profile.d/conda.sh \
    /ext3/conda3/etc/profile.d/conda.sh \
    ; do
    if [[ -f "$c" ]]; then conda_sh="$c"; break; fi
  done
fi

if [[ -n "$conda_sh" ]]; then
  # shellcheck disable=SC1090
  source "$conda_sh" || true
  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

# Common overlay layout: /ext3/<env>/bin/python
if ! command -v python >/dev/null 2>&1; then
  if [[ -d "/ext3/${CONDA_ENV}/bin" ]]; then
    export PATH="/ext3/${CONDA_ENV}/bin:${PATH}"
  fi
fi

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  alias python=python3
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in container PATH. Set CONDA_SH/CONDA_ENV or PYTHON_BIN." >&2
  exit 127
fi
'

singularity exec \
  "${fakeroot_args[@]}" \
  "${nv_args[@]}" \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  /bin/bash -lc "${activation_snippet}
${cmd}"
