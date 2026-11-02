#!/usr/bin/env bash
#SBATCH --job-name=spectral_age
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Spectral change from details to gist by age (pooled, within-story, within-story pooled) for one model
# and one saved batch.  CPU only.  Variables: MODEL_KEY (gpt_oss_20b|qwen3_8b), BATCH (folder name under
# data/processed/spectral_age/), DETAIL_VIEW (isolated|contextual), STEPS (check|pooled,within), RESELECT=1
# GLOBAL_DIR / DETAIL_DIR override the vector folders.
# (rebuild the batch on stories that have vectors; the batch actually used is written to <output>/selection/).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
BATCH="${BATCH:-batch_top800_ages2-8}"
DETAIL_VIEW="${DETAIL_VIEW:-isolated}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/representation/geometry/spectral_age/${MODEL_KEY}/${BATCH}_${DETAIL_VIEW}}"
EXTRA_ARGS=""
if [[ -n "${GLOBAL_DIR:-}" ]]; then EXTRA_ARGS="$EXTRA_ARGS --global-dir $GLOBAL_DIR"; fi
if [[ -n "${DETAIL_DIR:-}" ]]; then EXTRA_ARGS="$EXTRA_ARGS --detail-dir $DETAIL_DIR"; fi
EQUAL_ARG=""
if [[ "${EQUAL_CELLS:-0}" == "1" ]]; then EQUAL_ARG="--equal-cells"; fi
RESELECT_ARG=""
if [[ "${RESELECT:-1}" == "1" ]]; then
  RESELECT_ARG="--reselect-available"
fi

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.spectral_age.run \
  --batch-dir "data/processed/spectral_age/'"$BATCH"'" \
  --model-key "'"$MODEL_KEY"'" \
  --detail-view "'"$DETAIL_VIEW"'" \
  --layer "${LAYER:--1}" \
  --vector-key "${VECTOR_KEY:-mean_vectors}" \
  --steps "${STEPS:-pooled,within}" \
  --standardize \
  --splits "${SPLITS:-20}" \
  --permutations "${PERMUTATIONS:-20}" \
  --reduce-dim "${REDUCE_DIM:-256}" \
  --output-dir "'"$OUTPUT_DIR"'" '"$RESELECT_ARG $EXTRA_ARGS $EQUAL_ARG"''
