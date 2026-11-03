#!/usr/bin/env bash
#SBATCH --job-name=whole_vs_detail
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Overlap of each story's whole-story vector with its own detail vectors, by age (no PCA).  CPU only.
# Variables: MODEL_KEY (gpt_oss_20b|qwen3_8b), BATCH (folder under data/processed/spectral_age/), DETAIL_VIEW (isolated|contextual),
# VECTOR_KEY (mean_vectors|eos_vectors), LAYER (default -1), GLOBAL_DIR / DETAIL_DIR override the vector folders.
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
BATCH="${BATCH:-batch_top400_ages2-12}"
DETAIL_VIEW="${DETAIL_VIEW:-isolated}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/representation/geometry/whole_vs_detail/${MODEL_KEY}/${BATCH}_${DETAIL_VIEW}}"
EXTRA=""
if [[ -n "${GLOBAL_DIR:-}" ]]; then EXTRA="$EXTRA --global-dir $GLOBAL_DIR"; fi
if [[ -n "${DETAIL_DIR:-}" ]]; then EXTRA="$EXTRA --detail-dir $DETAIL_DIR"; fi
run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" -m analysis.representation.geometry.whole_vs_detail.run \
  --batch-dir data/processed/spectral_age/$BATCH --model-key $MODEL_KEY --detail-view $DETAIL_VIEW \
  --vector-key ${VECTOR_KEY:-mean_vectors} --layer ${LAYER:--1} --output-dir $OUTPUT_DIR $EXTRA"
