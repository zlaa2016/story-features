#!/usr/bin/env bash
#SBATCH --job-name=gist_structure
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# CPU job.  STEPS is a comma list of: calibrate, select, within, schema (default: select,within,schema).
# Variants (random selection, no topic matching, another model) go to their own OUTPUT_ROOT.
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
EXTRACTION_ROOT="${EXTRACTION_ROOT:-results/features/representation_extraction/causal_lm/detail_extraction}"
ISOLATED_DIR="${ISOLATED_DIR:-${EXTRACTION_ROOT}/${MODEL_KEY}_detail_controls}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/analysis/representation/geometry/gist_structure/${MODEL_KEY}}"
STEPS="${STEPS:-select,within,schema}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SUBSET_ARG=""
if [[ -n "${SUBSET_CSV:-}" ]]; then
  SUBSET_ARG="--subset-csv ${SUBSET_CSV}"
fi
MODULE=analysis.representation.geometry.gist_structure

step_enabled() { [[ ",${STEPS}," == *",$1,"* ]]; }

if step_enabled calibrate; then
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m '"$MODULE"'.calibrate \
    --per-band "${CALIBRATION_PER_BAND:-30}" --output-dir "'"$OUTPUT_ROOT"'/calibration"'
fi

if step_enabled select; then
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m '"$MODULE"'.select_stories \
    --isolated-dir "'"$ISOLATED_DIR"'" \
    --master-csv "${MASTER_CSV:-data/processed/master_stories.csv.gz}" \
    --bands "${BANDS:-2-3,4-5,6-7,8-12}" \
    --per-band "${PER_BAND:-100}" \
    --mode "${SELECT_MODE:-matched}" \
    --objective "${OBJECTIVE:-popular_coverage}" \
    --topic-quota "${TOPIC_QUOTA:-match}" \
    --topic-clusters "${TOPIC_CLUSTERS:-8}" \
    --min-units "${SELECT_MIN_UNITS:-30}" \
    --seed "${SEED:-42}" \
    --output-dir "'"$OUTPUT_ROOT"'/selection" '"$SUBSET_ARG"
fi

if step_enabled within; then
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m '"$MODULE"'.within_story \
    --isolated-dir "'"$ISOLATED_DIR"'" \
    --selected-csv "'"$OUTPUT_ROOT"'/selection/selected_stories.csv" \
    --layer "${LAYER:--1}" \
    --fixed-units "${FIXED_UNITS:-0,40}" \
    --min-units "${MIN_UNITS:-20}" \
    --workers "'"$WORKERS"'" \
    --bootstrap "${BOOTSTRAP:-300}" \
    --permutations "${PERMUTATIONS:-300}" \
    --seed "${SEED:-42}" \
    --output-dir "'"$OUTPUT_ROOT"'/within_story"'
fi

if step_enabled schema; then
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m '"$MODULE"'.schema_age \
    --isolated-dir "'"$ISOLATED_DIR"'" \
    --selected-csv "'"$OUTPUT_ROOT"'/selection/selected_stories.csv" \
    --layer "${LAYER:--1}" \
    --details-per-story "${DETAILS_PER_STORY:-40}" \
    --reduce-dim "${REDUCE_DIM:-96}" \
    --bootstrap "${SCHEMA_BOOTSTRAP:-60}" \
    --permutations "${SCHEMA_PERMUTATIONS:-100}" \
    --seed "${SEED:-42}" \
    --output-dir "'"$OUTPUT_ROOT"'/schema"'
fi
