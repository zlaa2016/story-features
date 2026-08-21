#!/usr/bin/env bash
#SBATCH --job-name=sae_intervene
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%6
#SBATCH --output=logs/mechanistic_interpretability/sae_intervene-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_intervene-%A_%a.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
LAYERS=(8 16 24)
LAYER_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SHARD=$((SLURM_ARRAY_TASK_ID / 3))
LAYER="${LAYERS[$LAYER_INDEX]}"
SAE_KIND="${SAE_KIND:-official}"
if [[ "$SAE_KIND" == "domain" ]]; then
  CHECKPOINT="mechanistic_interpretability/outputs/12_sparse_autoencoder/checkpoints/domain_sae_layer_$(printf '%02d' "$LAYER").pt"
  ANALYSIS_KIND="domain_trained"
else
  CHECKPOINT="mechanistic_interpretability/outputs/12_sparse_autoencoder/checkpoints/official_sae_layer_$(printf '%02d' "$LAYER").pt"
  ANALYSIS_KIND="official"
fi
run_in_container "PYTHONPATH=. $PYTHON_BIN mechanistic_interpretability/12_sparse_autoencoder/intervene.py --checkpoint $CHECKPOINT --feature-statistics mechanistic_interpretability/outputs/12_sparse_autoencoder/analysis/$ANALYSIS_KIND/layer_$(printf '%02d' "$LAYER")/feature_statistics.csv --shard-index $SHARD --num-shards 20 --output mechanistic_interpretability/outputs/12_sparse_autoencoder/${SAE_KIND}_interventions_layer_$(printf '%02d' "$LAYER")_shard_$(printf '%02d' "$SHARD").jsonl"
