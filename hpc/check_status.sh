#!/usr/bin/env bash
# Read-only status of the experiments on this machine: which outputs exist, how big, how recent, and your recent job states.
# Plain bash and standard tools only (no Python), so it runs on a login node as is:
#     bash hpc/check_status.sh
# It looks in the new layout (results/...) and the old one (mechanistic_interpretability/outputs/...), because outputs written
# before the reorganisation have not necessarily been copied yet. It deletes and moves nothing.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD="mechanistic_interpretability/outputs"
R="results"
EXTR="$R/features/representation_extraction/causal_lm/detail_extraction"

newest_date() {   # newest file modification date under a directory (GNU find on the cluster, BSD stat as a fallback)
  local out
  out="$(find -L "$1" -type f -printf '%TY-%Tm-%Td\n' 2>/dev/null | sort | tail -1)"
  [[ -z "$out" ]] && out="$(find -L "$1" -type f -exec stat -f '%Sm' -t '%Y-%m-%d' {} + 2>/dev/null | sort | tail -1)"
  echo "${out:--}"
}

step() {   # step <label> <new dir> <old dir> [required file patterns...]
  local label="$1" new="$2" old="$3"; shift 3
  local where dir n size when found="" pat count
  for where in new old; do
    dir="$new"; [[ "$where" == old ]] && dir="$old"
    [[ -n "$dir" && -d "$ROOT/$dir" ]] || continue
    n="$(find -L "$ROOT/$dir" -type f 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$n" -gt 0 ]] || continue
    size="$(du -shL "$ROOT/$dir" 2>/dev/null | cut -f1)"
    when="$(newest_date "$ROOT/$dir")"
    for pat in "$@"; do
      count="$(compgen -G "$ROOT/$dir/$pat" | wc -l | tr -d ' ')"
      found+=" $pat=$count"
    done
    printf '%-40s %-4s %7s %8s  %-10s %s\n' "$label" "$where" "$n" "$size" "$when" "$found"
    return
  done
  printf '%-40s %s\n' "$label" "MISSING"
}

printf '%-40s %-4s %7s %8s  %-10s %s\n' "step" "in" "files" "size" "newest" "required files found"
step "00 cohort" "$R/features/summaries/dataset_index" "$OLD/00_dataset_index" cohort.jsonl cohort_metadata.csv
step "02 extraction gpt_oss contextual" "$EXTR/gpt_oss_20b_global_current" "$OLD/02_representation_extraction/gpt_oss_20b_global_current" "contextual_detail_features_*.npz" "contextual_detail_metadata_*.csv"
step "02 extraction gpt_oss isolated" "$EXTR/gpt_oss_20b_detail_controls" "$OLD/02_representation_extraction/gpt_oss_20b_detail_controls" "isolated_detail_features_*.npz" "isolated_detail_metadata_*.csv"
step "02 extraction gpt_oss deletion" "$EXTR/gpt_oss_20b_detail_controls" "$OLD/02_representation_extraction/gpt_oss_20b_detail_controls" "deletion_features_*.npz" "deletion_metadata_*.csv"
step "02 extraction qwen3 contextual" "$EXTR/qwen3_8b_aug27_global" "$OLD/02_representation_extraction/qwen3_8b_aug27_global" "contextual_detail_features_*.npz"
step "02 extraction qwen3 isolated+deletion" "$EXTR/qwen3_8b_aug27_detail_controls" "$OLD/02_representation_extraction/qwen3_8b_aug27_detail_controls" "isolated_detail_features_*.npz" "deletion_features_*.npz"
step "03 native generation" "$R/features/summaries/native_generation" "$OLD/03_native_generation"
step "13 detail alignment (mpnet)" "$R/analysis/natural_language/detail_alignment" "$OLD/13_structured_detail_alignment" original_detail_retention.csv "shard_*/original_detail_retention.csv"
step "14 detail characteristics" "$R/analysis/natural_language/detail_characteristics" "$OLD/14_detail_characteristics"
step "15 mobius integration" "$R/analysis/natural_language/mobius_integration" "$OLD/15_mobius_detail_integration"
step "04 layerwise probes" "$R/analysis/representation/probes/layerwise_detail" "$OLD/04_layerwise_probes" story_split.csv
step "05 causal patching" "$R/analysis/representation/causal_patching" "$OLD/05_causal_patching"
step "06 activation steering" "$R/analysis/representation/steering/activation_steering" "$OLD/06_activation_steering"
step "07 router analysis" "$R/analysis/representation/router" "$OLD/07_router_analysis"
step "08 demixing" "$R/analysis/representation/demixing" "$OLD/08_demixing_normalization"
step "09 directional jacobian" "$R/analysis/representation/jacobian/directional_jacobian" "$OLD/09_directional_jacobian"
step "10 behavior vs representation" "$R/analysis/representation/behavior_vs_representation" "$OLD/10_behavior_representation_dissociation"
step "11 jacobian lens" "$R/analysis/representation/jacobian/jacobian_lens" "$OLD/11_jacobian_lens"
step "12 sparse autoencoder" "$R/analysis/representation/sparse_autoencoder" "$OLD/12_sparse_autoencoder"
step "16 compression geometry" "$R/analysis/representation/geometry/compression_geometry" "$OLD/16_causal_compression_geometry"
step "17 covariance geometry" "$R/analysis/representation/geometry/covariance" "$OLD/17_covariance_detail_geometry"
step "17 cross-model concordance" "$R/analysis/representation/geometry/cross_model_concordance" "$OLD/17_cross_model_concordance"
step "18 prefix probes" "$R/analysis/representation/probes/prefix_detail" "$OLD/18_prefix_detail_probes"
step "NEW span search, natural language" "$R/analysis/natural_language/detail_compression/span_search" "" summaries.csv details.csv
step "NEW span search, sentence level" "$R/analysis/natural_language/detail_compression/span_search_sentence" "" summaries.csv details.csv
step "NEW span search, contextual" "$R/analysis/representation/detail_compression/span_search/contextual_mean" "" summaries.csv details.csv
step "NEW span search, isolated" "$R/analysis/representation/detail_compression/span_search/isolated_mean" "" summaries.csv details.csv
step "NEW contextual-vs-isolated shift" "$R/analysis/representation/detail_compression/contextualization_shift" "" details.csv

echo
echo "Jobs in the last 45 days (by name and final state):"
if command -v sacct >/dev/null 2>&1; then
  sacct -X -u "${USER:-$(id -un)}" -S now-45days -n -P --format=JobName,State 2>/dev/null \
    | awk -F'|' '{split($2, s, " "); c[$1 "  " s[1]]++} END {for (k in c) printf "  %-44s %d\n", k, c[k]}' | sort
else
  echo "  (sacct not available here)"
fi
