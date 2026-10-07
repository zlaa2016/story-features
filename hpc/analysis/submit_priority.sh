#!/usr/bin/env bash

set -euo pipefail

SBATCH_ARGS_VALUE="${SBATCH_ARGS:-}"

submit() {
  local dependency="$1"
  local script="$2"
  local command=(sbatch --parsable)
  if [[ -n "$SBATCH_ARGS_VALUE" ]]; then
    read -r -a extra <<< "$SBATCH_ARGS_VALUE"
    command+=("${extra[@]}")
  fi
  if [[ -n "$dependency" ]]; then
    command+=(--dependency="afterok:$dependency")
  fi
  command+=("$script")
  "${command[@]}" | cut -d';' -f1
}

index_job="$(submit "" hpc/features/summaries/dataset_index/run.slurmsummaries/dataset_index/run.slurm)"
smoke_job="$(submit "$index_job" mechanistic_interpretability/01_smoke_test/run.slurm)"
extract_job="$(submit "$smoke_job" hpc/features/representation_extraction/causal_lm/detail_extraction/run.slurmrepresentation_extraction/causal_lm/detail_extraction/run.slurm)"
generation_job="$(submit "$smoke_job" hpc/features/summaries/native_generation/run.slurmsummaries/native_generation/run.slurm)"
probe_job="$(submit "$extract_job" hpc/analysis/representation/probes/layerwise_detail/run.slurmrepresentation/probes/layerwise_detail/run.slurm)"
router_job="$(submit "$extract_job" hpc/analysis/representation/router/run.slurmrepresentation/router/run.slurm)"
dissociation_job="$(submit "$extract_job" hpc/analysis/representation/behavior_vs_representation/run.slurmrepresentation/behavior_vs_representation/run.slurm)"
patch_job="$(submit "$probe_job" hpc/analysis/representation/causal_patching/run.slurmrepresentation/causal_patching/run.slurm)"
steer_job="$(submit "$probe_job" hpc/analysis/representation/steering/activation_steering/run.slurmrepresentation/steering/activation_steering/run.slurm)"
router_patch_job="$(submit "$probe_job" hpc/analysis/representation/router/causal_patch.slurmrepresentation/router/causal_patch.slurm)"
jacobian_job="$(submit "$probe_job" hpc/analysis/representation/jacobian/directional_jacobian/run.slurmrepresentation/jacobian/directional_jacobian/run.slurm)"
jlens_setup_job="$(submit "$index_job" hpc/analysis/representation/jacobian/jacobian_lens/setup.slurmrepresentation/jacobian/jacobian_lens/setup.slurm)"
jlens_fit_job="$(submit "$jlens_setup_job:$probe_job" hpc/analysis/representation/jacobian/jacobian_lens/fit.slurmrepresentation/jacobian/jacobian_lens/fit.slurm)"
jlens_merge_job="$(submit "$jlens_fit_job" hpc/analysis/representation/jacobian/jacobian_lens/merge.slurmrepresentation/jacobian/jacobian_lens/merge.slurm)"
jlens_apply_job="$(submit "$jlens_merge_job" hpc/analysis/representation/jacobian/jacobian_lens/apply.slurmrepresentation/jacobian/jacobian_lens/apply.slurm)"
jlens_analysis_job="$(submit "$jlens_apply_job" hpc/analysis/representation/jacobian/jacobian_lens/analyze.slurmrepresentation/jacobian/jacobian_lens/analyze.slurm)"
jlens_intervention_job="$(submit "$jlens_analysis_job" hpc/analysis/representation/jacobian/jacobian_lens/intervene.slurmrepresentation/jacobian/jacobian_lens/intervene.slurm)"
sae_collect_job="$(submit "$index_job" hpc/analysis/representation/sparse_autoencoder/collect.slurmrepresentation/sparse_autoencoder/collect.slurm)"
sae_official_job="$(submit "$index_job" hpc/analysis/representation/sparse_autoencoder/prepare_official.slurmrepresentation/sparse_autoencoder/prepare_official.slurm)"
sae_train_job="$(submit "$sae_collect_job:$probe_job" hpc/analysis/representation/sparse_autoencoder/train.slurmrepresentation/sparse_autoencoder/train.slurm)"
sae_analysis_job="$(submit "$sae_collect_job:$sae_official_job:$probe_job" hpc/analysis/representation/sparse_autoencoder/analyze.slurmrepresentation/sparse_autoencoder/analyze.slurm)"
sae_domain_analysis_job="$(submit "$sae_train_job" hpc/analysis/representation/sparse_autoencoder/analyze_domain.slurmrepresentation/sparse_autoencoder/analyze_domain.slurm)"
sae_intervention_job="$(submit "$sae_analysis_job" hpc/analysis/representation/sparse_autoencoder/intervene.slurmrepresentation/sparse_autoencoder/intervene.slurm)"
sae_survival_job=""
detail_alignment_job="$(submit "$index_job" hpc/analysis/natural_language/detail_alignment/run.slurmnatural_language/detail_alignment/run.slurm)"
detail_alignment_analysis_job="$(submit "$detail_alignment_job" hpc/analysis/natural_language/detail_alignment/analyze.slurmnatural_language/detail_alignment/analyze.slurm)"

normalization_job=""
if [[ -n "${RETENTION_CSV:-}" ]]; then
  normalization_job="$(submit "$probe_job" hpc/analysis/representation/demixing/run.slurmrepresentation/demixing/run.slurm)"
  sae_survival_job="$(submit "$sae_analysis_job" hpc/analysis/representation/sparse_autoencoder/survival.slurmrepresentation/sparse_autoencoder/survival.slurm)"
fi

cat <<EOF
dataset_index=$index_job
smoke=$smoke_job
representation_extraction=$extract_job
native_generation=$generation_job
layerwise_probes=$probe_job
router_analysis=$router_job
behavior_representation_dissociation=$dissociation_job
residual_patching=$patch_job
activation_steering=$steer_job
router_patching=$router_patch_job
directional_jacobian=$jacobian_job
jacobian_lens_fit=$jlens_fit_job
jacobian_lens_analysis=$jlens_analysis_job
jacobian_lens_intervention=$jlens_intervention_job
sae_collect=$sae_collect_job
sae_official=$sae_official_job
sae_domain_train=$sae_train_job
sae_official_analysis=$sae_analysis_job
sae_domain_analysis=$sae_domain_analysis_job
sae_intervention=$sae_intervention_job
sae_survival=${sae_survival_job:-not_submitted_set_RETENTION_CSV}
demixing_normalization=${normalization_job:-not_submitted_set_RETENTION_CSV}
structured_detail_alignment=$detail_alignment_job
structured_detail_analysis=$detail_alignment_analysis_job
EOF
