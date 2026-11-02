# Spectral change from details to gist, by age

One pipeline (`run.py`) for any model and any saved batch. It draws the age x share-of-variance heatmap of
"number of PCs needed" three ways:

| Panel | Rows | What it says |
|---|---|---|
| A pooled | one whole-story vector per story, stories at age a pooled | how many dimensions the stories of one age span (between stories) |
| B within | each story's own detail vectors, PCs counted per story, averaged over stories at age a | how fast one story's details collapse onto few components ("few gists") |
| C within-pooled | details centred on their own story's mean, pooled over stories at age a | the age's shared gist axes; half-split PC overlap between ages with a permuted-age null |

Also reported: participation ratio, entropy effective rank, spectral decay exponent, TwoNN intrinsic dimension
(`other_reductions_by_age.csv`, `story_profiles.csv`, `age_trend_within.csv` with size-adjusted partial rho against age).

## Subsets (saved first, `subsets.py`)
The unit is the **story** (`story_key`, e.g. `abc123__story02`); a video can hold several stories, each with its own text, details and vectors,
and a story inherits its video's stated range and view count. `data/processed/spectral_age/baseline_stories_span_lt7.csv`: 11,683 stories
(10,562 videos) whose stated range has `upper - lower <= 6` (under 7). Batches come from it: for each age, the most-viewed stories whose range
covers that age. The 800 batch has the same size at every age; the 400 batch keeps 400 wherever 400 exist (only age 12 is short, 374) and was built with `--keep-unequal`.
`run.py` keeps each age's own size by default; `EQUAL_CELLS=1` (`--equal-cells`) cuts all ages to the smallest as a sensitivity run (cell size changes k a little).

| Batch folder | Target | Cell size | Distinct |
|---|---|---|---|
| `batch_top800_ages2-8` | 800 per age, ages 2-8 | 800 | 2,134 stories (1,737 videos) |
| `batch_top400_ages2-12` | 400 per age, ages 2-12 | 400 (age 12: 374) | 1,738 stories (1,441 videos); only 374 narrow-range stories cover age 12 |

| `batch_flat_ages2-8_span_lt8` | flat design, ages 2-8, range under 8 (`upper - lower <= 7`) | 2,553 (same at every age) | 5,141 stories (4,542 videos) |

`batch_flat_*` is a different design: the largest set of stories whose coverage is *exactly equal* at every age (a story counts at every age its range covers),
solved as an integer program (`subsets.py flat`, `baseline_stories_span_lt8.csv`); a cell is every selected story covering the age. Stories of videos already queued
for extraction are preferred, then the most viewed. Its 2,521 videos outside the first extraction list are in `extraction_videos_flat_extra.csv`.
With `--reselect-available` the flat design is solved again on the stories that have vectors, so the run uses the largest flat set the vectors allow.

Each holds `subset_stories.csv`, `membership.csv` (age, story_key, rank), `summary.json`. Rebuild with `python -m analysis.representation.geometry.spectral_age.subsets baseline|select ...`
(`baseline` needs the local `data/exports/`).

## Run on the cluster
```bash
git fetch origin && git checkout compress && git pull --ff-only
# 1. which batch stories have vectors (prints and writes vector_coverage.json)
STEPS=check RESELECT=0 BATCH=batch_top800_ages2-8 MODEL_KEY=gpt_oss_20b sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/spectral_age/run.slurm
# 2. the real runs (RESELECT=1 rebuilds the batch on stories that have vectors and saves it under <output>/selection/)
for B in batch_top800_ages2-8 batch_top400_ages2-12 batch_flat_ages2-8_span_lt8; do for M in gpt_oss_20b qwen3_8b; do
  BATCH=$B MODEL_KEY=$M sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/spectral_age/run.slurm
done; done
```
Outputs in `results/analysis/representation/geometry/spectral_age/<model>/<batch>_<detail view>/`: `fig_spectrum_by_age.png`
(the three heatmaps), `pooled/`, `within_<view>/`, `selection/` (the exact stories the run used), `run_args.json`.
`DETAIL_VIEW=contextual` repeats B and C on in-context detail states.

## Extracting vectors for batch videos the cluster lacks
`cohort.jsonl` was built with an age filter and an older story set, so it misses many batch stories. `extraction_videos.csv` freezes the 2,602 videos
that hold every batch story (so shard boundaries of a running submission stay valid). `build_supplement_cohort.py` (run locally,
needs `data/exports/`) writes cohort-format records for every story of those videos to `data/processed/spectral_age/supplement_cohort.jsonl.gz`.
`filter_cohort.py` then writes `spectral_age_cohort*.jsonl`: the cohort's own batch records first (same order as before), then supplement
records for videos the cohort lacks; `--exclude-dirs` removes videos a model already has vectors for; videos found nowhere go to
`*.missing_videos.csv`. `submit_extraction.sh` submits two chained array jobs (whole-story + in-context details with `extract_global.py`, isolated
details with `extract_detail_controls.py`) into `<root>/<model>_spectral_global/` and `<root>/<model>_spectral_detail_controls/`.
`run.py` reads every folder listed in `MODEL_DIRS` that exists (first folder wins when a video is in several).

```bash
# Qwen, continuing the first 10 shards (records 0-935 were already submitted from the cohort alone); new shard ids start at 100
MODEL_KEY=qwen3_8b COHORT_NAME=spectral_age_cohort_part2.jsonl START_OFFSET=936 SHARD_OFFSET=100 bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
# GPT-OSS: only the batch videos that have no vectors yet
R=results/features/representation_extraction/causal_lm/detail_extraction
MODEL_KEY=gpt_oss_20b COHORT_NAME=spectral_age_cohort_gpt_oss.jsonl EXCLUDE_DIRS="$R/gpt_oss_20b_global_current" bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
```
Extra videos of the flat batch (2,521 videos, 2,784 stories; shard ids from 300 so nothing collides):
```bash
D=data/processed/spectral_age; R=results/features/representation_extraction/causal_lm/detail_extraction
MODEL_KEY=qwen3_8b VIDEOS_CSV=$D/extraction_videos_flat_extra.csv SUPPLEMENT=$D/supplement_cohort_flat_extra.jsonl.gz COHORT_NAME=spectral_age_cohort_flat_extra.jsonl SHARD_OFFSET=300 bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
MODEL_KEY=gpt_oss_20b VIDEOS_CSV=$D/extraction_videos_flat_extra.csv SUPPLEMENT=$D/supplement_cohort_flat_extra.jsonl.gz COHORT_NAME=spectral_age_cohort_flat_extra_gpt_oss.jsonl SHARD_OFFSET=300 EXCLUDE_DIRS="$R/gpt_oss_20b_global_current" bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
```
Qwen3-8B reads up to 32k tokens (`MAX_TOKENS=32768`, longer stories skipped and reported).

## Caution: detail granularity differs between sources
Stories already on the cluster were segmented by the older detail prompt (about one sentence per detail); the supplement's segments come from the
newer run (about 3 sentences per detail). Within-story spectra (B, C) depend on the number of units, so the reported `kn` (k / (units - 1)), the size-adjusted
trend tests and `story_profiles.csv` (`n_units`) must be read with that in mind; panel A (whole-story vectors) does not depend on segmentation.
