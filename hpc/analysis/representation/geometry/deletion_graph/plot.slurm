#!/usr/bin/env bash
#SBATCH --job-name=deletion_graph_plot
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# One picture per story: the directed deletion-coupling graph with its source hubs and integrators marked.  CPU only.
# Variables: MODEL_KEY, BATCH, VIEW (mean|last), DISTANCE (cos|rel), LAYER, N_STORIES (default 12 most viewed with >= 8 details),
# STORIES (comma-separated story keys instead), EDGES (excess|raw), TOP_K (marked per kind).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
BATCH="${BATCH:-batch_top100_ages2-12}"
ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
INPUT="${INPUT_DIR:-$ROOT/${MODEL_KEY}_deletion_graph_${BATCH}}"
OUT="${OUTPUT_DIR:-results/analysis/representation/geometry/deletion_graph/${MODEL_KEY}/${BATCH}_graphs}"
COHORTS="results/features/summaries/dataset_index/cohort.jsonl"
for FILE in "$ROOT"/spectral_age_cohort*.jsonl; do
  [[ -f "$FILE" ]] && COHORTS="$COHORTS $FILE"
done
run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" -m analysis.representation.geometry.deletion_graph.plot_story_graphs \
  --input-dir $INPUT --batch data/processed/spectral_age/$BATCH --output-dir $OUT --cohort $COHORTS \
  --view ${VIEW:-mean} --distance ${DISTANCE:-cos} ${LAYER:+--layer $LAYER} --n-stories ${N_STORIES:-12} \
  --edges ${EDGES:-excess} --top-k ${TOP_K:-3} ${STORIES:+--stories $STORIES}"
