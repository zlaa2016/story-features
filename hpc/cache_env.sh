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

# One runtime folder for every cache, so a single directory holds them all. Model downloads stay where they already are
# (HF_HOME defaults to /scratch/$USER/hf) so existing downloads are reused.
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-$SCRATCH_ROOT/story-features-runtime}"
export RUNTIME_CACHE_ROOT

_to_scratch XDG_CACHE_HOME "$RUNTIME_CACHE_ROOT/xdg-cache"
_to_scratch XDG_DATA_HOME "$RUNTIME_CACHE_ROOT/xdg-data"
_to_scratch HF_HOME "$SCRATCH_ROOT/hf"
_to_scratch TRANSFORMERS_CACHE "$HF_HOME"
_to_scratch HUGGINGFACE_HUB_CACHE "$HF_HOME/hub"
_to_scratch TORCH_HOME "$RUNTIME_CACHE_ROOT/torch"
_to_scratch SENTENCE_TRANSFORMERS_HOME "$RUNTIME_CACHE_ROOT/sentence_transformers"
_to_scratch PIP_CACHE_DIR "$RUNTIME_CACHE_ROOT/pip"
_to_scratch MPLCONFIGDIR "$RUNTIME_CACHE_ROOT/matplotlib"
_to_scratch NUMBA_CACHE_DIR "$RUNTIME_CACHE_ROOT/numba"
_to_scratch TRITON_CACHE_DIR "$RUNTIME_CACHE_ROOT/triton"
_to_scratch CUDA_CACHE_PATH "$RUNTIME_CACHE_ROOT/cuda"
_to_scratch PYTHONPYCACHEPREFIX "$RUNTIME_CACHE_ROOT/pycache"
_to_scratch IPYTHONDIR "$RUNTIME_CACHE_ROOT/ipython"
_to_scratch JUPYTER_CONFIG_DIR "$RUNTIME_CACHE_ROOT/jupyter-config"
_to_scratch JUPYTER_DATA_DIR "$RUNTIME_CACHE_ROOT/jupyter-data"
_to_scratch JUPYTER_RUNTIME_DIR "$RUNTIME_CACHE_ROOT/jupyter-runtime"
_to_scratch TMPDIR "$RUNTIME_CACHE_ROOT/tmp"
_to_scratch SINGULARITY_CACHEDIR "$RUNTIME_CACHE_ROOT/singularity-cache"
_to_scratch SINGULARITY_TMPDIR "$RUNTIME_CACHE_ROOT/singularity-tmp"
_to_scratch APPTAINER_CACHEDIR "$SINGULARITY_CACHEDIR"
_to_scratch APPTAINER_TMPDIR "$SINGULARITY_TMPDIR"
unset -f _to_scratch
