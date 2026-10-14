#!/usr/bin/env bash
#SBATCH --job-name=convert_features
#SBATCH --output=logs/mechanistic_interpretability/%x-%j.out
#SBATCH --error=logs/mechanistic_interpretability/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
source hpc/slurm_env.sh

# The layerwise probes (04) and behavior-vs-representation (10) read results/.../detail_extraction/<MODEL_KEY>/features_*.npz.
# Build those from the contextual shards the global extraction already wrote, instead of a second GPU pass.
# CPU only. Existing converted shards are kept; pass OVERWRITE=1 to redo them.
EXTRACTION="results/features/representation_extraction/causal_lm/detail_extraction"
MODEL_KEY="${MODEL_KEY:-gpt_oss_20b}"
GLOBAL_DIR="${GLOBAL_DIR:-$EXTRACTION/${MODEL_KEY}_global_current}"
export MODEL_KEY GLOBAL_DIR EXTRACTION
compgen -G "$GLOBAL_DIR/contextual_detail_features_*.npz" >/dev/null || { echo "No contextual shards under $GLOBAL_DIR" >&2; exit 1; }

run_in_container 'PYTHONPATH=src "$PYTHON_BIN" src/features/representation_extraction/causal_lm/detail_extraction/convert_contextual_to_detail_features.py \
  --input-dir "$GLOBAL_DIR" \
  --output-dir "$EXTRACTION/$MODEL_KEY" \
  --model-key "$MODEL_KEY" \
  --cohort results/features/summaries/dataset_index/cohort.jsonl \
  ${OVERWRITE:+--overwrite}'
