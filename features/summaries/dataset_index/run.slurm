#!/usr/bin/env bash
#SBATCH --job-name=mech_index
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
source shared/slurm_env.sh

REPAIR_FRAGMENT_ARG=""
if [[ "${REPAIR_TITLE_FRAGMENTS:-0}" == "1" ]]; then
  REPAIR_FRAGMENT_ARG="--repair-title-fragments"
fi
export REPAIR_FRAGMENT_ARG

run_in_container '"$PYTHON_BIN" -m features.summaries.dataset_index.build_index \
  --output-dir results/features/summaries/dataset_index \
  --source-age-min "${SOURCE_AGE_MIN:-2}" \
  --source-age-max "${SOURCE_AGE_MAX:-10}" \
  --minimum-age "${TARGET_AGE_MIN:-2}" \
  --maximum-age "${TARGET_AGE_MAX:-18}" \
  --max-summaries "${MAX_SUMMARIES:-3}" \
  ${REPAIR_FRAGMENT_ARG:-} && \
"$PYTHON_BIN" -m features.summaries.dataset_index.validate \
  --input-dir results/features/summaries/dataset_index \
  --minimum-age "${TARGET_AGE_MIN:-2}" \
  --maximum-age "${TARGET_AGE_MAX:-18}"'
