#!/usr/bin/env bash
#SBATCH --job-name=prefix_probe_stats
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.sh
export USE_NV=0
export FAKEROOT="${FAKEROOT:-0}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/scratch/$USER/story-features-runtime}"
export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg-cache"
export TMPDIR="$RUNTIME_CACHE_ROOT/tmp"
export HF_HOME="${HF_HOME:-$RUNTIME_CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="$RUNTIME_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$RUNTIME_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$RUNTIME_CACHE_ROOT/cuda"
export NUMBA_CACHE_DIR="$RUNTIME_CACHE_ROOT/numba"
export PYTHONPYCACHEPREFIX="$RUNTIME_CACHE_ROOT/pycache"
export MPLCONFIGDIR="$RUNTIME_CACHE_ROOT/matplotlib"
export IPYTHONDIR="$RUNTIME_CACHE_ROOT/ipython"
export JUPYTER_CONFIG_DIR="$RUNTIME_CACHE_ROOT/jupyter-config"
export JUPYTER_DATA_DIR="$RUNTIME_CACHE_ROOT/jupyter-data"
export PIP_CACHE_DIR="$RUNTIME_CACHE_ROOT/pip"
export JUPYTER_PATH="$PWD/hpc/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
mkdir -p \
  "$XDG_CACHE_HOME" "$TMPDIR" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$IPYTHONDIR" \
  "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$PIP_CACHE_DIR"

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
PREFIX_MODE="${PREFIX_MODE:-native}"
INPUT_DIR="${INPUT_DIR:-results/analysis/representation/probes/prefix_detail/${MODEL_KEY}_${PREFIX_MODE}}"
RETENTION_DIR="${RETENTION_DIR:-results/analysis/natural_language/detail_alignment}"
OUTPUT_DIR="${ANALYSIS_DIR:-output/prefix_detail_probes/${MODEL_KEY}_${PREFIX_MODE}}"
export MODEL_KEY PREFIX_MODE INPUT_DIR RETENTION_DIR OUTPUT_DIR

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -c \
  "import numpy, pandas, matplotlib, scipy, statsmodels, patsy; print(\"analysis dependencies OK\")"'

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" \
  src/analysis/representation/probes/prefix_detail/analyze.py \
  --input-dir "$INPUT_DIR" \
  --retention-dir "$RETENTION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pooling-equivalence-margin "${POOLING_EQUIVALENCE_MARGIN:-0.02}"'

echo "Completed analysis: $OUTPUT_DIR"

export STORY_FEATURES_ROOT="$PWD"
if [[ "$OUTPUT_DIR" = /* ]]; then
  export PREFIX_PROBE_ANALYSIS_DIR="$OUTPUT_DIR"
else
  export PREFIX_PROBE_ANALYSIS_DIR="$PWD/$OUTPUT_DIR"
fi
run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/analysis/representation/probes/gpt_oss_prefix_detail_probe_analysis.ipynb \
  --output "$PREFIX_PROBE_ANALYSIS_DIR/gpt_oss_prefix_detail_probe_analysis.executed.ipynb" \
  --ExecutePreprocessor.kernel_name=story-features-direct \
  --ExecutePreprocessor.timeout=0'

echo "Completed notebook: $OUTPUT_DIR/gpt_oss_prefix_detail_probe_analysis.executed.ipynb"
