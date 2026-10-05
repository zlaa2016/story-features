#!/usr/bin/env bash
#SBATCH --job-name=prefix_probe_stats
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
export USE_NV=0
export FAKEROOT="${FAKEROOT:-0}"
export PYDEPS="${PYDEPS:-/scratch/$USER/story-features-pydeps}"
export PYTHONPATH="$PYDEPS:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}"
mkdir -p "$PYDEPS" "$MPLCONFIGDIR"

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
PREFIX_MODE="${PREFIX_MODE:-native}"
INPUT_DIR="${INPUT_DIR:-mechanistic_interpretability/outputs/18_prefix_detail_probes/${MODEL_KEY}_${PREFIX_MODE}}"
RETENTION_DIR="${RETENTION_DIR:-mechanistic_interpretability/outputs/13_structured_detail_alignment}"
OUTPUT_DIR="${ANALYSIS_DIR:-output/prefix_detail_probes/${MODEL_KEY}_${PREFIX_MODE}}"
export MODEL_KEY PREFIX_MODE INPUT_DIR RETENTION_DIR OUTPUT_DIR

if ! run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" -c "import statsmodels, patsy"' >/dev/null 2>&1; then
  run_in_container '"$PYTHON_BIN" -m pip install --target "'"$PYDEPS"'" --no-deps \
    "statsmodels==0.14.4" "patsy==1.0.1"'
fi

run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" -c \
  "import numpy, pandas, matplotlib, scipy, statsmodels, patsy; print(\"analysis dependencies OK\")"'

run_in_container 'PYTHONPATH="'"$PYDEPS"':$PWD" "$PYTHON_BIN" \
  mechanistic_interpretability/18_prefix_detail_probes/analyze.py \
  --input-dir "$INPUT_DIR" \
  --retention-dir "$RETENTION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pooling-equivalence-margin "${POOLING_EQUIVALENCE_MARGIN:-0.02}"'

echo "Completed analysis: $OUTPUT_DIR"
