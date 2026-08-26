#!/usr/bin/env bash
#SBATCH --job-name=repr_dissociation
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh

MODEL_A_NAME="${MODEL_A_NAME:-gpt_oss_20b}"
MODEL_B_NAME="${MODEL_B_NAME:-qwen3_8b}"
MODEL_A_DIR="${MODEL_A_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_A_NAME}_detail_controls}"
MODEL_B_DIR="${MODEL_B_DIR:-mechanistic_interpretability/outputs/02_representation_extraction/${MODEL_B_NAME}_detail_controls}"
RETENTION_CSV="${RETENTION_CSV:-mechanistic_interpretability/outputs/13_structured_detail_alignment/original_detail_retention.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-mechanistic_interpretability/outputs/17_covariance_detail_geometry/${MODEL_A_NAME}_vs_${MODEL_B_NAME}}"

run_in_container 'PYTHONPATH=. "$PYTHON_BIN" -m mechanistic_interpretability.17_covariance_detail_geometry.compare_models \
  --model-a-dir "'"$MODEL_A_DIR"'" \
  --model-b-dir "'"$MODEL_B_DIR"'" \
  --model-a-name "'"$MODEL_A_NAME"'" \
  --model-b-name "'"$MODEL_B_NAME"'" \
  --retention-csv "'"$RETENTION_CSV"'" \
  --output-dir "'"$OUTPUT_DIR"'" \
  --label-column "${LABEL_COLUMN:-retained_structured}" \
  --pca-components "${PCA_COMPONENTS:-64}" \
  --decoder-folds "${DECODER_FOLDS:-5}" \
  --permutation-iterations "${PERMUTATION_ITERATIONS:-200}" \
  --seed "${SEED:-42}"'
