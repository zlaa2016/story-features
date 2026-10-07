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
export PYDEPS="${PYDEPS:-/scratch/$USER/story-features-pydeps}"
export PYTHONPATH="$PYDEPS:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}"
export IPYTHONDIR="${IPYTHONDIR:-/scratch/$USER/.ipython-story-features}"
export JUPYTER_PATH="$PWD/env/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
mkdir -p "$PYDEPS" "$MPLCONFIGDIR" "$IPYTHONDIR"

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
PREFIX_MODE="${PREFIX_MODE:-native}"
INPUT_DIR="${INPUT_DIR:-results/analysis/representation/probes/prefix_detail/${MODEL_KEY}_${PREFIX_MODE}}"
RETENTION_DIR="${RETENTION_DIR:-results/analysis/natural_language/detail_alignment}"
OUTPUT_DIR="${ANALYSIS_DIR:-output/prefix_detail_probes/${MODEL_KEY}_${PREFIX_MODE}}"
export MODEL_KEY PREFIX_MODE INPUT_DIR RETENTION_DIR OUTPUT_DIR

if ! run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" -c "import statsmodels, patsy"' >/dev/null 2>&1; then
  run_in_container '"$PYTHON_BIN" -m pip install --target "'"$PYDEPS"'" --no-deps \
    "statsmodels==0.14.4" "patsy==1.0.1"'
fi

run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" -c \
  "import numpy, pandas, matplotlib, scipy, statsmodels, patsy; print(\"analysis dependencies OK\")"'

run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" \
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
run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/analysis/representation/probes/gpt_oss_prefix_detail_probe_analysis.ipynb \
  --output "$PREFIX_PROBE_ANALYSIS_DIR/gpt_oss_prefix_detail_probe_analysis.executed.ipynb" \
  --ExecutePreprocessor.kernel_name=story-features-direct \
  --ExecutePreprocessor.timeout=0'

echo "Completed notebook: $OUTPUT_DIR/gpt_oss_prefix_detail_probe_analysis.executed.ipynb"
