#!/usr/bin/env bash
#SBATCH --job-name=deletion_graph_units
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Event graphs against detail graphs of the same stories (CPU).  Needs analyze.slurm run once with UNITS=events and once without.
# Variables: MODEL_KEY, BATCH, VIEW (mean|last), DISTANCE (cos|rel).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
BATCH="${BATCH:-batch_events_ages2-12}"
VIEW="${VIEW:-mean}"
DISTANCE="${DISTANCE:-cos}"
BASE="results/analysis/representation/geometry/deletion_graph/${MODEL_KEY}"
run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" -m analysis.representation.geometry.deletion_graph.compare_units \
  --events $BASE/${BATCH}_events_${VIEW}_${DISTANCE} --details $BASE/${BATCH}_${VIEW}_${DISTANCE} \
  --batch data/processed/spectral_age/$BATCH --output-dir $BASE/${BATCH}_events_vs_details_${VIEW}_${DISTANCE}"
