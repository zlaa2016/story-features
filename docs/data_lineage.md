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
2. **Sentence-level summaries in a detail-level cohort.** Everything except the legacy review samples is segmented into details,
   apart from the sentence-level file, which is intentional. Measured segment sizes:

   | File | Segments per record (median) | Words per segment (median) |
   |---|---|---|
   | Originals (consolidated) | 64 | 7 |
   | Summaries, cascade (`api_detail`) | 24 | 6 |
   | Summaries, combined "current analysis", labelled `api_detail` | 23 | 6 |
   | Summaries, combined "current analysis", unlabelled | 20 | 7 |
   | Summaries, `local_sentence` | 11 | 14 |

   `local_sentence` is sentence splitting, not detail segmentation (about twice the words per segment), so ratios that divide by the
   number of summary details are not comparable. In the cohort config it contributes 148 summaries that exist nowhere else
   (about 2.6%). It is a deliberate second measurement of compression: how many full-text details each summary *sentence* covers. It is run on its own
   (`span_search_sentence.slurm`, no vectors) and is no longer a cohort input (`sentence_summary_jsonl` in `default_paths.json`).
   The detail-level runs skip any summary labelled `local_sentence` (`--exclude-methods`). The cohort currently on the cluster and the
   GPT-OSS vectors still contain 148 such summaries until the cohort is rebuilt.
3. **Unlabelled records.** 2,540 of the 6,997 records in the "current analysis" file have no `segmentation_method`. Their segment
   sizes (7 words) match detail segmentation, and 2,422 of them also appear in the cascade file, but only 324 of those with the same
   number of segments, so they come from a different (earlier) segmentation run.

Deciding which summary file is canonical is still open.

## Legacy: event-segmented

`data/legacy/review_samples_event_segments/` (383 markdown files) is the earlier event-segmented output. It is read only by:

- `data_pipeline/cleaning/master_dataset/build_master_dataset.py` (`review_events_json`, `has_events`)
- `features/eda/analyze_text_features.py` (event and concept features) and `features/eda/find_corpus_aoa_cutoffs.py`
- the DistilBERT events-unit vectors and `analysis/representation/age_aoa_direction/`, `analysis/representation/steering/distilbert/`
- the concept notebooks in `notebooks/features/eda/` (`aoa_word2vec_concepts`, `concepts_age_unified`, `quick_feature_viz`,
  `eda_age_topics_popularity_reviews`, `word_concepts_to_story_age`)

It is not read by the cohort, the span search, the GPT-OSS extraction, the contextual/isolated shift or the MiniLM `pm3` analyses.
