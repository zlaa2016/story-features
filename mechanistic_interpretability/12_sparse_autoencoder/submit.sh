#!/usr/bin/env bash
set -euo pipefail
ARGS=(${SBATCH_ARGS:-})
collect=$(sbatch --parsable "${ARGS[@]}" mechanistic_interpretability/12_sparse_autoencoder/collect.slurm | cut -d';' -f1)
official=$(sbatch --parsable "${ARGS[@]}" mechanistic_interpretability/12_sparse_autoencoder/prepare_official.slurm | cut -d';' -f1)
train=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$collect" mechanistic_interpretability/12_sparse_autoencoder/train.slurm | cut -d';' -f1)
analyze=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$collect:$official" mechanistic_interpretability/12_sparse_autoencoder/analyze.slurm | cut -d';' -f1)
domain_analyze=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$train" mechanistic_interpretability/12_sparse_autoencoder/analyze_domain.slurm | cut -d';' -f1)
intervene=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$analyze" mechanistic_interpretability/12_sparse_autoencoder/intervene.slurm | cut -d';' -f1)
survival="not_submitted_set_RETENTION_CSV"
if [[ -n "${RETENTION_CSV:-}" ]]; then survival=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$analyze" mechanistic_interpretability/12_sparse_autoencoder/survival.slurm | cut -d';' -f1); fi
printf 'sae_collect=%s\nsae_official=%s\nsae_domain_train=%s\nsae_official_analyze=%s\nsae_domain_analyze=%s\nsae_intervene=%s\nsae_survival=%s\n' "$collect" "$official" "$train" "$analyze" "$domain_analyze" "$intervene" "$survival"
