#!/usr/bin/env bash
#SBATCH --job-name=sae_survival
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/mechanistic_interpretability/sae_survival-%A_%a.out
#SBATCH --error=logs/mechanistic_interpretability/sae_survival-%A_%a.err

set -euo pipefail
source mechanistic_interpretability/shared/slurm_env.sh
: "${RETENTION_CSV:?Set RETENTION_CSV to original_detail_retention.csv}"
LAYERS=(8 16 24); LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
SAE_KIND="${SAE_KIND:-official}"
if [[ "$SAE_KIND" == "domain" ]]; then
  CHECKPOINT="mechanistic_interpretability/outputs/12_sparse_autoencoder/checkpoints/domain_sae_layer_$(printf '%02d' "$LAYER").pt"
  ANALYSIS_KIND="domain_trained"
else
  CHECKPOINT="mechanistic_interpretability/outputs/12_sparse_autoencoder/checkpoints/official_sae_layer_$(printf '%02d' "$LAYER").pt"
  ANALYSIS_KIND="official"
fi
run_in_container "PYTHONPATH=. $PYTHON_BIN mechanistic_interpretability/12_sparse_autoencoder/survival.py --checkpoint $CHECKPOINT --retention-csv '$RETENTION_CSV' --output mechanistic_interpretability/outputs/12_sparse_autoencoder/analysis/$ANALYSIS_KIND/layer_$(printf '%02d' "$LAYER")/detail_survival.csv"
