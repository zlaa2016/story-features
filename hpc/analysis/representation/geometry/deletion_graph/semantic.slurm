#!/usr/bin/env bash
#SBATCH --job-name=deletion_graph_semantic
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Semantic-relatedness graph (cosine of existing detail vectors) against the deletion-coupling graph of the same stories.  CPU only; nothing is re-extracted.
# Variables: MODEL_KEY, BATCH, DETAIL_VIEW (isolated|contextual), DETAIL_LAYER (default -1), STANDARDIZE (global|none), VIEW (mean|last), DISTANCE (cos|rel),
# DETAIL_DIRS (space-separated folders; default: the model's folders from spectral_age/run.py MODEL_DIRS under the extraction root).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
BATCH="${BATCH:-batch_top100_ages2-12}"
VIEW="${VIEW:-mean}"
DISTANCE="${DISTANCE:-cos}"
DETAIL_VIEW="${DETAIL_VIEW:-isolated}"
ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
INPUT="${INPUT_DIR:-$ROOT/${MODEL_KEY}_deletion_graph_${BATCH}}"
OUT="${OUTPUT_DIR:-results/analysis/representation/geometry/deletion_graph/${MODEL_KEY}/${BATCH}_semantic_${DETAIL_VIEW}_${VIEW}_${DISTANCE}}"
DIRS="${DETAIL_DIRS:-$ROOT/${MODEL_KEY}_spectral_detail_controls $ROOT/${MODEL_KEY}_detail_controls}"
run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" -m analysis.representation.geometry.deletion_graph.semantic_graph \
  --input-dir $INPUT --detail-dir $DIRS --batch data/processed/spectral_age/$BATCH --output-dir $OUT \
  --detail-view $DETAIL_VIEW --detail-layer ${DETAIL_LAYER:--1} --standardize ${STANDARDIZE:-global} --view $VIEW --distance $DISTANCE"
