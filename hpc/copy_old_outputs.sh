#!/usr/bin/env bash
# Copy outputs written before the reorganisation into the new results/ layout.
# Safe to re-run: existing files are never overwritten and nothing is deleted.
# Run from the repository root on each machine that holds old outputs.
set -euo pipefail

move() {
  local old="" new=""
  if [[ -d "" ]]; then
    mkdir -p "" && rsync -a --ignore-existing "/" "/" && echo "copied  -> "
  fi
}

move "mechanistic_interpretability/outputs/00_dataset_index" "results/features/summaries/dataset_index"
move "mechanistic_interpretability/outputs/03_native_generation" "results/features/summaries/native_generation"
move "mechanistic_interpretability/outputs/02_representation_extraction/analyze_detail_compression" "results/analysis/representation/detail_compression/analyze_detail_compression"
move "mechanistic_interpretability/outputs/02_representation_extraction/analyze_global" "results/analysis/representation/detail_compression/analyze_global"
move "mechanistic_interpretability/outputs/02_representation_extraction" "results/features/representation_extraction/causal_lm/detail_extraction"
move "mechanistic_interpretability/outputs/13_structured_detail_alignment" "results/analysis/natural_language/detail_alignment"
move "mechanistic_interpretability/outputs/14_detail_characteristics" "results/analysis/natural_language/detail_characteristics"
move "mechanistic_interpretability/outputs/15_mobius_detail_integration" "results/analysis/natural_language/mobius_integration"
move "mechanistic_interpretability/outputs/04_layerwise_probes" "results/analysis/representation/probes/layerwise_detail"
move "mechanistic_interpretability/outputs/18_prefix_detail_probes" "results/analysis/representation/probes/prefix_detail"
move "mechanistic_interpretability/outputs/05_causal_patching" "results/analysis/representation/causal_patching"
move "mechanistic_interpretability/outputs/06_activation_steering" "results/analysis/representation/steering/activation_steering"
move "mechanistic_interpretability/outputs/07_router_analysis" "results/analysis/representation/router"
move "mechanistic_interpretability/outputs/08_demixing_normalization" "results/analysis/representation/demixing"
move "mechanistic_interpretability/outputs/09_directional_jacobian" "results/analysis/representation/jacobian/directional_jacobian"
move "mechanistic_interpretability/outputs/11_jacobian_lens" "results/analysis/representation/jacobian/jacobian_lens"
move "mechanistic_interpretability/outputs/10_behavior_representation_dissociation" "results/analysis/representation/behavior_vs_representation"
move "mechanistic_interpretability/outputs/12_sparse_autoencoder" "results/analysis/representation/sparse_autoencoder"
move "mechanistic_interpretability/outputs/16_causal_compression_geometry" "results/analysis/representation/geometry/compression_geometry"
move "mechanistic_interpretability/outputs/17_covariance_detail_geometry" "results/analysis/representation/geometry/covariance"
move "mechanistic_interpretability/outputs/17_cross_model_concordance" "results/analysis/representation/geometry/cross_model_concordance"
move "pipeline_output" "results/features/summaries/runs"
move "artifacts/age_aoa_direction_experiments" "results/analysis/representation/age_aoa_direction/age_aoa_direction_experiments"
move "artifacts/aoa_lexicon_concepts" "results/features/eda/aoa_lexicon_concepts"
move "artifacts/aoa_word2vec" "results/features/eda/aoa_word2vec"
move "artifacts/concept_age" "results/features/eda/concept_age"
move "artifacts/concept_age_unified" "results/features/eda/concept_age_unified"
move "artifacts/corpus_aoa_cutoffs" "results/features/eda/corpus_aoa_cutoffs"
move "artifacts/corpus_token_distributions_cleaned" "results/features/eda/corpus_token_distributions_cleaned"
move "artifacts/decoder_rewrites" "results/analysis/representation/steering/decoder_rewrite/decoder_rewrites"
move "artifacts/distilbert_vectors" "results/features/representation_extraction/encoder/distilbert_vectors"
move "artifacts/embedding_steering_experiments" "results/analysis/representation/steering/distilbert/embedding_steering_experiments"
move "artifacts/event_features_batch1" "results/features/eda/event_features_batch1"
move "artifacts/feature_baseline" "results/analysis/natural_language/feature_baseline/feature_baseline"
move "artifacts/feature_baseline_batch1_review_samples" "results/analysis/natural_language/feature_baseline/feature_baseline_batch1_review_samples"
move "artifacts/finetuned_age_classifier" "results/analysis/natural_language/age_classifier/finetuned_age_classifier"
move "artifacts/gpt_oss_vectors" "results/features/representation_extraction/causal_lm/gpt_oss_vectors"
move "artifacts/layerwise_regression_probe" "results/analysis/representation/probes/layerwise_regression_probe"
move "artifacts/linear_probe" "results/analysis/representation/probes/linear_probe"
move "artifacts/perplexity" "results/analysis/natural_language/perplexity"
move "artifacts/probe_concept_mean_event_words" "results/features/eda/probe_concept_mean_event_words"
move "artifacts/rewrite_eval" "results/analysis/natural_language/rewrite_evaluation/rewrite_eval"
move "artifacts/steer" "results/analysis/representation/steering/distilbert/steer"
move "artifacts/text_features" "results/features/eda/text_features"
