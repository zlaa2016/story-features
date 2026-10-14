# Legacy data

`review_samples_event_segments/` holds 383 markdown files from the earlier **event-segmented** pipeline. Each
file lists a story's events; `build_master_dataset.py` parses them into `review_events_json` and `has_events`.

They are not detail-segmented and are not read by the current experiments (cohort, span search, GPT-OSS
extraction, contextual/isolated shift, MiniLM `pm3`). Only the older feature and probe work uses them:
`analyze_text_features.py`, `find_corpus_aoa_cutoffs.py`, the DistilBERT events-unit vectors,
`run_age_aoa_direction_experiments.py`, `run_embedding_steering_experiments.py`, and the concept notebooks in
`notebooks/features/eda/`. See `docs/data_lineage.md`.
