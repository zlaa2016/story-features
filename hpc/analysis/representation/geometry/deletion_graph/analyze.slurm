#!/usr/bin/env bash
#SBATCH --job-name=deletion_graph_analyze
#SBATCH --account=torch_pr_494_general
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Graph measures per story and their change with age.  CPU only.  Variables: MODEL_KEY, BATCH, VIEW (mean|last), DISTANCE (cos|rel), LAYER (number; default the last saved).
set -euo pipefail
source hpc/slurm_env.sh

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
BATCH="${BATCH:-batch_top100_ages2-12}"
VIEW="${VIEW:-mean}"
DISTANCE="${DISTANCE:-cos}"
INPUT="${INPUT_DIR:-results/features/representation_extraction/causal_lm/detail_extraction/${MODEL_KEY}_deletion_graph_${BATCH}}"
OUT="${OUTPUT_DIR:-results/analysis/representation/geometry/deletion_graph/${MODEL_KEY}/${BATCH}_${VIEW}_${DISTANCE}${LAYER:+_layer$LAYER}}"
run_in_container "PYTHONPATH=src \"\$PYTHON_BIN\" -m analysis.representation.geometry.deletion_graph.analyze \
  --input-dir $INPUT --batch data/processed/spectral_age/$BATCH --output-dir $OUT --view $VIEW --distance $DISTANCE ${LAYER:+--layer $LAYER}"
