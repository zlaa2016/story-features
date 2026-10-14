#!/usr/bin/env bash
# Keep every cache, temp file and tool config on /scratch so $HOME stays under its quota.
#
# Sourced by hpc/slurm_env.sh and hpc/run_in_singularity.sh, so every job and every container run
# gets it. Safe to source twice. A variable you already set to a path outside $HOME is left alone;
# one that is unset, or points into $HOME, is redirected to $SCRATCH_ROOT.
#
# Singularity binds your real $HOME into the container by default, so anything that falls back to
# ~/.cache, ~/.local or ~/.jupyter inside a job writes to the home quota. Setting these variables
# stops the common ones. For everything else, set HOME_ON_SCRATCH=1 (see run_in_singularity.sh).

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER:-$(id -un)}}"
export SCRATCH_ROOT

_to_scratch() {
  local name="$1" path="$2" current="${!1:-}"
  if [[ -z "$current" || "${current}" == "${HOME:-/nonexistent-home}"* ]]; then
    export "$name=$path"
  fi
  mkdir -p "${!name}" 2>/dev/null || true
}

_to_scratch XDG_CACHE_HOME "$SCRATCH_ROOT/.cache"
_to_scratch XDG_DATA_HOME "$SCRATCH_ROOT/.local/share"
_to_scratch HF_HOME "$SCRATCH_ROOT/hf"
_to_scratch TRANSFORMERS_CACHE "$HF_HOME"
_to_scratch TORCH_HOME "$SCRATCH_ROOT/.cache/torch"
_to_scratch SENTENCE_TRANSFORMERS_HOME "$SCRATCH_ROOT/.cache/sentence_transformers"
_to_scratch PIP_CACHE_DIR "$SCRATCH_ROOT/.cache/pip"
_to_scratch MPLCONFIGDIR "$SCRATCH_ROOT/.cache/matplotlib"
_to_scratch NUMBA_CACHE_DIR "$SCRATCH_ROOT/.cache/numba"
_to_scratch TRITON_CACHE_DIR "$SCRATCH_ROOT/.cache/triton"
_to_scratch IPYTHONDIR "$SCRATCH_ROOT/.ipython-story-features"
_to_scratch JUPYTER_DATA_DIR "$SCRATCH_ROOT/.local/share/jupyter"
_to_scratch JUPYTER_RUNTIME_DIR "$SCRATCH_ROOT/.local/share/jupyter/runtime"
_to_scratch TMPDIR "$SCRATCH_ROOT/tmp"
_to_scratch SINGULARITY_CACHEDIR "$SCRATCH_ROOT/.singularity/cache"
_to_scratch SINGULARITY_TMPDIR "$SCRATCH_ROOT/.singularity/tmp"
_to_scratch APPTAINER_CACHEDIR "$SINGULARITY_CACHEDIR"
_to_scratch APPTAINER_TMPDIR "$SINGULARITY_TMPDIR"
unset -f _to_scratch
