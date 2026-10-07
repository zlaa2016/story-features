#!/usr/bin/env bash
#SBATCH --job-name=cov_detail_geom
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
DELETION_DIR="${DELETION_DIR:-results/features/representation_extraction/causal_lm/detail_extraction/${MODEL_KEY}_detail_controls}"
RETENTION_CSV="${RETENTION_CSV:-results/analysis/natural_language/detail_alignment/original_detail_retention.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/representation/geometry/covariance/${MODEL_KEY}}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.covariance.analyze \
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
