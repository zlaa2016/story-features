#!/usr/bin/env bash

set -euo pipefail
SBATCH_ARGS_VALUE="${SBATCH_ARGS:---account=torch_pr_494_general}"
RUN_SAE="${RUN_SAE:-0}"

submit() {
  local dependency="$1"
  local script="$2"
  shift 2
  local command=(sbatch --parsable)
  read -r -a extra <<< "$SBATCH_ARGS_VALUE"
  command+=("${extra[@]}")
  [[ -n "$dependency" ]] && command+=(--dependency="afterok:${dependency}")
  command+=("$@" "$script")
  "${command[@]}" | cut -d';' -f1
}

index_job=$(submit "" hpc/features/summaries/dataset_index/run.slurm)
gpt_align_job=$(submit "$index_job" hpc/analysis/natural_language/detail_alignment/run.slurmnatural_language/detail_alignment/run.slurm)
gpt_align_analysis_job=$(submit "$gpt_align_job" hpc/analysis/natural_language/detail_alignment/analyze.slurmnatural_language/detail_alignment/analyze.slurm)
detail_characteristics_job=$(submit "$gpt_align_analysis_job" hpc/analysis/natural_language/detail_characteristics/run.slurmnatural_language/detail_characteristics/run.slurm)
mobius_job=$(submit "$index_job" hpc/analysis/natural_language/mobius_integration/run.slurmnatural_language/mobius_integration/run.slurm)
mobius_analysis_job=$(submit "$mobius_job" hpc/analysis/natural_language/mobius_integration/analyze.slurmnatural_language/mobius_integration/analyze.slurm)

declare -A native_analysis_jobs
declare -A causal_analysis_jobs
declare -A jacobian_jobs
declare -A probe_jobs
declare -A dissociation_jobs
declare -A patch_jobs
for model in qwen3_8b gpt_oss_20b; do
  export_args="--export=ALL,MODEL_KEY=${model}"
  smoke=$(submit "$index_job" mechanistic_interpretability/01_smoke_test/run.slurm "$export_args")
  extract=$(submit "$smoke" hpc/features/representation_extraction/causal_lm/detail_extraction/run.slurmrepresentation_extraction/causal_lm/detail_extraction/run.slurm "$export_args")
  native=$(submit "$smoke" hpc/features/summaries/native_generation/run.slurm "$export_args")
  probe=$(submit "${extract}:${gpt_align_analysis_job}" hpc/analysis/representation/probes/layerwise_detail/run.slurmrepresentation/probes/layerwise_detail/run.slurm "$export_args")
  probe_jobs[$model]="$probe"
  dissociation_jobs[$model]=$(submit "$extract" hpc/analysis/representation/behavior_vs_representation/run.slurmrepresentation/behavior_vs_representation/run.slurm "$export_args")
  patch_jobs[$model]=$(submit "$probe" hpc/analysis/representation/causal_patching/run.slurmrepresentation/causal_patching/run.slurm "$export_args")
  jacobian_jobs[$model]=$(submit "$probe" hpc/analysis/representation/jacobian/directional_jacobian/run.slurmrepresentation/jacobian/directional_jacobian/run.slurm "$export_args")

  native_prep=$(submit "$native" hpc/features/summaries/native_generation/prepare.slurm "$export_args")
  native_input="results/features/summaries/native_generation/${model}/alignment_records"
  native_output="results/features/summaries/native_generation/${model}/alignment"
  native_align=$(submit "$native_prep" hpc/analysis/natural_language/detail_alignment/run.slurmnatural_language/detail_alignment/run.slurm \
    "--export=ALL,MODEL_KEY=${model},DETAIL_INPUT_BASE=${native_input},DETAIL_OUTPUT_DIR=${native_output}")
  native_analysis=$(submit "$native_align" hpc/analysis/natural_language/detail_alignment/analyze.slurmnatural_language/detail_alignment/analyze.slurm \
    "--export=ALL,MODEL_KEY=${model},DETAIL_OUTPUT_DIR=${native_output}")
  native_analysis_jobs[$model]="$native_analysis"

  steering=$(submit "$probe" hpc/analysis/representation/steering/activation_steering/run.slurmrepresentation/steering/activation_steering/run.slurm "$export_args")
  causal_prep=$(submit "$steering" hpc/analysis/representation/geometry/compression_geometry/prepare.slurmrepresentation/geometry/compression_geometry/prepare.slurm "$export_args")
  causal_align=$(submit "$causal_prep" hpc/analysis/representation/geometry/compression_geometry/align.slurmrepresentation/geometry/compression_geometry/align.slurm "$export_args")
  causal_analysis=$(submit "$causal_align" hpc/analysis/representation/geometry/compression_geometry/analyze.slurmrepresentation/geometry/compression_geometry/analyze.slurm "$export_args")
  causal_analysis_jobs[$model]="$causal_analysis"
done

router_analysis_job=$(submit "${probe_jobs[gpt_oss_20b]}" hpc/analysis/representation/router/run.slurmrepresentation/router/run.slurm \
  "--export=ALL,MODEL_KEY=gpt_oss_20b")
router_patch_job=$(submit "${probe_jobs[gpt_oss_20b]}" hpc/analysis/representation/router/causal_patch.slurmrepresentation/router/causal_patch.slurm \
  "--export=ALL,MODEL_KEY=gpt_oss_20b")

jlens_setup_job=$(submit "$index_job" hpc/analysis/representation/jacobian/jacobian_lens/setup.slurmrepresentation/jacobian/jacobian_lens/setup.slurm)
jlens_fit_job=$(submit "${jlens_setup_job}:${probe_jobs[qwen3_8b]}" hpc/analysis/representation/jacobian/jacobian_lens/fit.slurmrepresentation/jacobian/jacobian_lens/fit.slurm)
jlens_merge_job=$(submit "$jlens_fit_job" hpc/analysis/representation/jacobian/jacobian_lens/merge.slurmrepresentation/jacobian/jacobian_lens/merge.slurm)
jlens_apply_job=$(submit "$jlens_merge_job" hpc/analysis/representation/jacobian/jacobian_lens/apply.slurmrepresentation/jacobian/jacobian_lens/apply.slurm)
jlens_analysis_job=$(submit "$jlens_apply_job" hpc/analysis/representation/jacobian/jacobian_lens/analyze.slurmrepresentation/jacobian/jacobian_lens/analyze.slurm)
jlens_intervention_job=$(submit "$jlens_analysis_job" hpc/analysis/representation/jacobian/jacobian_lens/intervene.slurmrepresentation/jacobian/jacobian_lens/intervene.slurm)

sae_analysis_job="skipped"
sae_domain_analysis_job="skipped"
sae_official_intervention_job="skipped"
sae_domain_intervention_job="skipped"
sae_official_survival_job="skipped"
sae_domain_survival_job="skipped"
if [[ "$RUN_SAE" == "1" ]]; then
  sae_collect_job=$(submit "$index_job" hpc/analysis/representation/sparse_autoencoder/collect.slurmrepresentation/sparse_autoencoder/collect.slurm)
  sae_official_job=$(submit "$index_job" hpc/analysis/representation/sparse_autoencoder/prepare_official.slurmrepresentation/sparse_autoencoder/prepare_official.slurm)
  sae_train_job=$(submit "${sae_collect_job}:${probe_jobs[qwen3_8b]}" hpc/analysis/representation/sparse_autoencoder/train.slurmrepresentation/sparse_autoencoder/train.slurm)
  sae_analysis_job=$(submit "${sae_collect_job}:${sae_official_job}:${probe_jobs[qwen3_8b]}" hpc/analysis/representation/sparse_autoencoder/analyze.slurmrepresentation/sparse_autoencoder/analyze.slurm)
  sae_domain_analysis_job=$(submit "$sae_train_job" hpc/analysis/representation/sparse_autoencoder/analyze_domain.slurmrepresentation/sparse_autoencoder/analyze_domain.slurm)
  sae_official_intervention_job=$(submit "$sae_analysis_job" hpc/analysis/representation/sparse_autoencoder/intervene.slurmrepresentation/sparse_autoencoder/intervene.slurm \
    "--export=ALL,SAE_KIND=official")
  sae_domain_intervention_job=$(submit "$sae_domain_analysis_job" hpc/analysis/representation/sparse_autoencoder/intervene.slurmrepresentation/sparse_autoencoder/intervene.slurm \
    "--export=ALL,SAE_KIND=domain")
  sae_official_survival_job=$(submit "${sae_analysis_job}:${gpt_align_analysis_job}" hpc/analysis/representation/sparse_autoencoder/survival.slurmrepresentation/sparse_autoencoder/survival.slurm \
    "--export=ALL,RETENTION_CSV=results/analysis/natural_language/detail_alignment/original_detail_retention.csv")
  sae_domain_survival_job=$(submit "${sae_domain_analysis_job}:${gpt_align_analysis_job}" hpc/analysis/representation/sparse_autoencoder/survival.slurmrepresentation/sparse_autoencoder/survival.slurm \
    "--export=ALL,SAE_KIND=domain,RETENTION_CSV=results/analysis/natural_language/detail_alignment/original_detail_retention.csv")
fi

concordance_dependency="${native_analysis_jobs[qwen3_8b]}:${native_analysis_jobs[gpt_oss_20b]}:${gpt_align_analysis_job}"
concordance_job=$(submit "$concordance_dependency" hpc/analysis/representation/geometry/cross_model_concordance/run.slurmrepresentation/geometry/cross_model_concordance/run.slurm)

cat <<EOF
dataset_index=$index_job
gpt_detail_alignment=$gpt_align_job
gpt_detail_analysis=$gpt_align_analysis_job
detail_characteristics=$detail_characteristics_job
mobius_integration=$mobius_job
mobius_analysis=$mobius_analysis_job
qwen_native_analysis=${native_analysis_jobs[qwen3_8b]}
gpt_oss_native_analysis=${native_analysis_jobs[gpt_oss_20b]}
qwen_causal_geometry=${causal_analysis_jobs[qwen3_8b]}
gpt_oss_causal_geometry=${causal_analysis_jobs[gpt_oss_20b]}
qwen_directional_jacobian=${jacobian_jobs[qwen3_8b]}
gpt_oss_directional_jacobian=${jacobian_jobs[gpt_oss_20b]}
qwen_behavior_representation_dissociation=${dissociation_jobs[qwen3_8b]}
gpt_oss_behavior_representation_dissociation=${dissociation_jobs[gpt_oss_20b]}
qwen_residual_patching=${patch_jobs[qwen3_8b]}
gpt_oss_residual_patching=${patch_jobs[gpt_oss_20b]}
gpt_oss_router_analysis=$router_analysis_job
gpt_oss_router_patching=$router_patch_job
jacobian_lens_analysis=$jlens_analysis_job
jacobian_lens_intervention=$jlens_intervention_job
sae_official_analysis=$sae_analysis_job
sae_domain_analysis=$sae_domain_analysis_job
sae_official_intervention=$sae_official_intervention_job
sae_domain_intervention=$sae_domain_intervention_job
sae_official_detail_survival=$sae_official_survival_job
sae_domain_detail_survival=$sae_domain_survival_job
cross_model_concordance=$concordance_job
EOF
