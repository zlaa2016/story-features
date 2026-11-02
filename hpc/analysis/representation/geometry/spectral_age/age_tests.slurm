#!/usr/bin/env bash
#SBATCH --job-name=spectral_age_tests
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Curve-level within-story age test on finished runs (reads story_profiles.csv, no vectors).  CPU only.
# Variables: MODEL_KEY, DETAIL_VIEW, BATCHES (space separated batch names), PERMUTATIONS.
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
DETAIL_VIEW="${DETAIL_VIEW:-isolated}"
BATCHES="${BATCHES:-batch_top800_ages2-8 batch_top400_ages2-12 batch_flat_ages2-8_span_lt8}"
ROOT="results/analysis/representation/geometry/spectral_age/${MODEL_KEY}"

for B in $BATCHES; do
  echo "== $B"
  run_in_container 'PYTHONPATH=src "$PYTHON_BIN" -m analysis.representation.geometry.spectral_age.age_tests \
    --profiles "'"$ROOT/${B}_${DETAIL_VIEW}/within_${DETAIL_VIEW}/story_profiles.csv"'" \
    --permutations "${PERMUTATIONS:-2000}"'
done
