#!/usr/bin/env bash
#SBATCH --job-name=within_story_compress
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
EXTRACTION_ROOT="${EXTRACTION_ROOT:-results/features/representation_extraction/causal_lm/detail_extraction}"
ISOLATED_DIR="${ISOLATED_DIR:-${EXTRACTION_ROOT}/${MODEL_KEY}_detail_controls}"
CONTEXTUAL_DIR="${CONTEXTUAL_DIR:-${EXTRACTION_ROOT}/${MODEL_KEY}_global_current}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/representation/geometry/within_story_compression/${MODEL_KEY}}"
SUBSET_ARG=""
if [[ -n "${SUBSET_CSV:-}" ]]; then
  SUBSET_ARG="--subset-csv ${SUBSET_CSV}"
fi
EQUALIZE_ARG=""
if [[ "${EQUALIZE:-0}" == "1" ]]; then
  EQUALIZE_ARG="--equalize"
fi

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.within_story_compression.analyze \
  --isolated-dir "'"$ISOLATED_DIR"'" \
  --contextual-dir "'"$CONTEXTUAL_DIR"'" \
  --views "${VIEWS:-isolated,contextual}" \
  --layer "${LAYER:--1}" \
  --age-min "${AGE_MIN:-2}" \
  --age-max "${AGE_MAX:-12}" \
  --per-age "${PER_AGE:-200}" \
  --n-assignments "${N_ASSIGNMENTS:-5}" \
  --min-units "${MIN_UNITS:-20}" \
  --max-units "${MAX_UNITS:-300}" \
  --null-draws "${NULL_DRAWS:-10}" \
  --position-components "${POSITION_COMPONENTS:-3}" \
  --permutations "${PERMUTATIONS:-2000}" \
  --bootstrap "${BOOTSTRAP:-1000}" \
  --seed "${SEED:-42}" \
  --output-dir "'"$OUTPUT_DIR"'" \
  '"$SUBSET_ARG $EQUALIZE_ARG"''
