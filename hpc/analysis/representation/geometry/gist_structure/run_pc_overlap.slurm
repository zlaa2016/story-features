#!/usr/bin/env bash
#SBATCH --job-name=pc_overlap_by_age
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# PC overlap between ages at several variance levels, from the GPT-OSS whole-story vectors of the
# original stories (global_features_*.npz), on the flat 2-12 list.  CPU only.
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
EXTRACTION_ROOT="${EXTRACTION_ROOT:-results/features/representation_extraction/causal_lm/detail_extraction}"
GLOBAL_DIR="${GLOBAL_DIR:-${EXTRACTION_ROOT}/${MODEL_KEY}_global_current}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/representation/geometry/gist_structure/pc_overlap/${MODEL_KEY}}"
FLAT_CSV="${FLAT_CSV:-data/processed/flat_videos_D_2-12.csv}"

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.gist_structure.pc_overlap_by_age \
  --source gpt_oss_global \
  --global-dir "'"$GLOBAL_DIR"'" \
  --vector-key "${VECTOR_KEY:-mean_vectors}" \
  --layer "${LAYER:--1}" \
  --flat-csv "'"$FLAT_CSV"'" \
  --standardize \
  --modes "${MODES:-multi}" \
  --per-age "${PER_AGE:-500}" \
  --splits "${SPLITS:-60}" \
  --permutations "${PERMUTATIONS:-40}" \
  --output-dir "'"$OUTPUT_DIR"'"'
