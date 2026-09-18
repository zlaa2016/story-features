#!/usr/bin/env bash
#SBATCH --job-name=gptoss_repr_align
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p output/gpt_oss_detail_representation_alignment

export SIF=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
export OVERLAY="/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro"
export CONDA_ENV=story-features
export PYTHON_BIN=/ext3/story-features/bin/python

export PYDEPS="/scratch/$USER/story-features-pydeps"
export MPLCONFIGDIR="/scratch/$USER/.cache/matplotlib"
mkdir -p "$PYDEPS" "$MPLCONFIGDIR"
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

export GPT_OSS_CONTEXT_DIR="${GPT_OSS_CONTEXT_DIR:-$PWD/mechanistic_interpretability/outputs/02_representation_extraction/gpt_oss_20b_global_current}"
export GPT_OSS_ISOLATED_DIR="${GPT_OSS_ISOLATED_DIR:-$PWD/mechanistic_interpretability/outputs/02_representation_extraction/gpt_oss_20b_detail_controls}"
export MPNET_ALIGNMENT_DIR="${MPNET_ALIGNMENT_DIR:-$PWD/mechanistic_interpretability/outputs/13_structured_detail_alignment}"

test -d "$GPT_OSS_CONTEXT_DIR"
test -d "$GPT_OSS_ISOLATED_DIR"
test -d "$MPNET_ALIGNMENT_DIR"
compgen -G "$GPT_OSS_CONTEXT_DIR/contextual_detail_features_*.npz" >/dev/null || {
  echo "No contextual-detail feature shards under $GPT_OSS_CONTEXT_DIR" >&2
  exit 1
}
compgen -G "$GPT_OSS_ISOLATED_DIR/isolated_detail_features_*.npz" >/dev/null || {
  echo "No isolated-detail feature shards under $GPT_OSS_ISOLATED_DIR" >&2
  exit 1
}
if [[ ! -f "$MPNET_ALIGNMENT_DIR/original_detail_retention.csv" ]] \
  && ! compgen -G "$MPNET_ALIGNMENT_DIR/shard_*/original_detail_retention.csv" >/dev/null; then
  echo "No MPNet original-detail retention table under $MPNET_ALIGNMENT_DIR" >&2
  exit 1
fi

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
    --execute notebooks/gpt_oss_detail_representation_alignment.ipynb \
    --output "$(pwd)/output/gpt_oss_detail_representation_alignment/gpt_oss_detail_representation_alignment.executed.ipynb" \
    --ExecutePreprocessor.timeout=0'

echo "Completed: output/gpt_oss_detail_representation_alignment/gpt_oss_detail_representation_alignment.executed.ipynb"
