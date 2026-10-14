# Data lineage: which files feed which experiments

Paths below are relative to `pipeline_output/` in the original local layout. In this repository they live under
`results/features/summaries/runs/` (the old `pipeline_output/` folder was moved there). Counts were measured on the
local copies; the cluster copies were not checked.

## Detail-segmented pipeline (current)

| Contents | File | Records |
|---|---|---|
| Main cleaned text and original details | `transcript_cleanup_detail_age_manifest/transcript_cleanup_detail_age_manifest_detail_segments.jsonl` (+ `cleaned_txt/`) | 6,406 |
| Additional cleaned originals and details (backfill) | `customer_age_2_10_remaining_cheapest/detail_cleanup/customer_age_2_10_remaining_detail_detail_segments.jsonl` | 1,109 (1,076 stories) |
| Consolidated originals for the customer-age analysis | `customer_age_target_or_source_2_10_preprocess/original_detail_segments_source_or_target_2_10_customer_age.jsonl` | 3,806 stories |
| Combined API-segmented summaries ("current analysis") | `current_detail_retention_stats/combined_api_detail_segments_summaries.jsonl` | 6,997 summaries, 3,824 stories |
| Original large summary-generation batch (gpt-5.4-mini) | `age_steering_pm3_customer_discrete_resume_gpt-5.4-mini_20260817_080237/summary_outputs.jsonl` | 5,166 |
| Cascade-generated missing summaries | `customer_age_target_or_source_2_10_cascade_summary_detail_min1/generated_summary_outputs.jsonl` | 332 |

Producing scripts: `data_pipeline/cleaning/transcript_cleanup/run_transcript_cleanup_detail_segmentation.py` (cleaned text and
original details; imports helper functions from `run_transcript_cleanup_segmentation.py`),
`features/summaries/cascade_summary_detail_pipeline.py` and `features/summaries/detail_segment_age_steering_subset.py`
(segment generated summaries into details).

## What the cohort is built from

`shared/config/default_paths.json` lists the inputs of `features/summaries/dataset_index/build_index.py`:

- **Originals:** the consolidated file above plus `customer_age_2_10_min_one_backfill/combined_original_detail_segments.jsonl`.
  Both give the same 3,806 stories, so the originals are consistent with the table.
- **Summary details:** two files, `customer_age_target_or_source_2_10_cascade_summary_detail_min1/detail_segments_summaries.jsonl`
  (6,879 summaries, `api_detail` segmentation) and
  `age_steering_pm3_customer_discrete_resume_.../detail_segments_current_subset_local_sentence/detail_segments_summaries.jsonl`
  (5,474 summaries, `local_sentence` segmentation).

### Known mismatches (not yet resolved)

1. **The cohort does not read the "current analysis" summary file.** Against it, the config's two files have 69 summaries
   it lacks, and it has 39 they lack.
2. **Mixed segmentation units.** 148 summaries in the config exist only as `local_sentence` segments. For summaries present in both
   files, `local_sentence` has about 0.46 times as many segments as `api_detail` (median), so ratios that divide by the number of
   summary details are not comparable across the two. Use `SEGMENTATION_METHOD=api_detail` when selecting the compression subset
   (`select_flat_subset.slurm`), or `--require-method api_detail` on the script. The script prints how many summaries each method
   contributes before filtering.
3. **Unlabelled records.** 2,540 of the 6,997 records in the "current analysis" file have no `segmentation_method`, so their
   method is not recorded.

Deciding which summary file is canonical is still open.

## Legacy: event-segmented

`data/legacy/review_samples_event_segments/` (383 markdown files) is the earlier event-segmented output. It is read only by:

- `data_pipeline/cleaning/master_dataset/build_master_dataset.py` (`review_events_json`, `has_events`)
- `features/eda/analyze_text_features.py` (event and concept features) and `features/eda/find_corpus_aoa_cutoffs.py`
- the DistilBERT events-unit vectors and `analysis/representation/age_aoa_direction/`, `analysis/representation/steering/distilbert/`
- the concept notebooks in `notebooks/features/eda/` (`aoa_word2vec_concepts`, `concepts_age_unified`, `quick_feature_viz`,
  `eda_age_topics_popularity_reviews`, `word_concepts_to_story_age`)

It is not read by the cohort, the span search, the GPT-OSS extraction, the contextual/isolated shift or the MiniLM `pm3` analyses.
