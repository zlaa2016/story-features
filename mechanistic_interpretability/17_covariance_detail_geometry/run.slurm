#!/usr/bin/env bash
#SBATCH --job-name=cov_detail_geom
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
DELETION_DIR="${DELETION_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_KEY}_detail_controls}"
RETENTION_CSV="${RETENTION_CSV:-mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-mechanistic_interpretability/outputs/17_covariance_detail_geometry/${MODEL_KEY}}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.17_covariance_detail_geometry.analyze \
  --deletion-dir "'"$DELETION_DIR"'" \
  --retention-csv "'"$RETENTION_CSV"'" \
  --output-dir "'"$OUTPUT_DIR"'" \
  --label-column "${LABEL_COLUMN:-retained_structured}" \
  --poolings "${POOLINGS:-eos,mean}" \
  --geometry-modes "${GEOMETRY_MODES:-raw,direction}" \
  --groupings "${GROUPINGS:-summary_condition,summary_condition+target_age}" \
  --pca-components "${PCA_COMPONENTS:-64}" \
  --pca-fit-rows "${PCA_FIT_ROWS:-20000}" \
  --subspace-rank "${SUBSPACE_RANK:-10}" \
  --minimum-per-class "${MINIMUM_PER_CLASS:-30}" \
  --decoder-folds "${DECODER_FOLDS:-5}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-200}" \
  --permutation-iterations "${PERMUTATION_ITERATIONS:-200}" \
  --seed "${SEED:-42}"'
