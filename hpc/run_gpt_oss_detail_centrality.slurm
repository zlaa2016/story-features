#!/usr/bin/env bash
#SBATCH --job-name=gptoss_centrality
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output/gpt_oss_detail_centrality

# Container and the existing project Python environment.
export SIF=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
export OVERLAY="/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro"
export CONDA_ENV=story-features
export PYTHON_BIN=/ext3/story-features/bin/python

# Writable locations: the image and overlay above are read-only.
export PYDEPS="/scratch/$USER/story-features-pydeps"
export MPLCONFIGDIR="/scratch/$USER/.cache/matplotlib"
mkdir -p "$PYDEPS" "$MPLCONFIGDIR"
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

# Install only the package missing from the image. Existing numpy, scipy,
# pandas, and patsy remain supplied by the project environment.
if ! ./hpc/run_in_singularity.sh \
  '"$PYTHON_BIN" -c "import statsmodels"' >/dev/null 2>&1; then
  ./hpc/run_in_singularity.sh \
    '"$PYTHON_BIN" -m pip install --target "$PYDEPS" --no-deps "statsmodels==0.14.4"'
fi

./hpc/run_in_singularity.sh \
  '"$PYTHON_BIN" -c "import sys, numpy, pandas, matplotlib, scipy, statsmodels, patsy; print(sys.executable); print(\"dependencies OK; statsmodels\", statsmodels.__version__)"'

./hpc/run_in_singularity.sh \
  '"$PYTHON_BIN" -m jupyter nbconvert \
    --to notebook \
    --execute notebooks/gpt_oss_detail_centrality_retention.ipynb \
    --output "$(pwd)/output/gpt_oss_detail_centrality/gpt_oss_detail_centrality_retention.executed.ipynb" \
    --ExecutePreprocessor.timeout=0'

echo "Completed: output/gpt_oss_detail_centrality/gpt_oss_detail_centrality_retention.executed.ipynb"
