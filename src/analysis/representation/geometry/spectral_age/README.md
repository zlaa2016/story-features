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
`data/processed/spectral_age/baseline_span_lt7.csv`: 10,562 videos whose stated range has `upper - lower <= 6` (under 7).
Batches are built from it: for each age, the most-viewed videos whose range covers that age; every age is cut to the smallest age.

| Batch folder | Target | Cell size | Note |
|---|---|---|---|
| `batch_top800_ages2-8` | 800 per age, ages 2-8 | 800 | 2,135 distinct videos |
| `batch_top400_ages2-12` | 400 per age, ages 2-12 | 341 | only 384 / 341 narrow-range videos cover ages 11 / 12 |

Each holds `subset_videos.csv`, `membership.csv`, `summary.json`. Rebuild: `python -m analysis.representation.geometry.spectral_age.subsets select --per-age 800 --age-min 2 --age-max 8 --output-dir ...`
(needs the baseline; the local `data/exports/` is only needed for `baseline`).

## Run on the cluster
```bash
git fetch origin && git checkout compress && git pull --ff-only
# 1. which batch videos have vectors (prints and writes vector_coverage.json)
STEPS=check RESELECT=0 BATCH=batch_top800_ages2-8 MODEL_KEY=gpt_oss_20b sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/spectral_age/run.slurm
# 2. the real runs (RESELECT=1 rebuilds the batch on videos that have vectors and saves it under <output>/selection/)
for B in batch_top800_ages2-8 batch_top400_ages2-12; do for M in gpt_oss_20b qwen3_8b; do
  BATCH=$B MODEL_KEY=$M sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/spectral_age/run.slurm
done; done
```
Outputs in `results/analysis/representation/geometry/spectral_age/<model>/<batch>_<detail view>/`: `fig_spectrum_by_age.png`
(the three heatmaps), `pooled/`, `within_<view>/`, `selection/` (the exact videos the run used), `run_args.json`.
`DETAIL_VIEW=contextual` repeats B and C on in-context detail states.

## Extracting vectors for batch videos the cluster lacks
`cohort.jsonl` was built with an age filter and an older story set, so it misses many batch videos. `build_supplement_cohort.py` (run locally,
needs `data/exports/`) writes cohort-format records for every batch video to `data/processed/spectral_age/supplement_cohort.jsonl.gz`.
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
Qwen3-8B reads up to 32k tokens (`MAX_TOKENS=32768`, longer stories skipped and reported).

## Caution: detail granularity differs between sources
Stories already on the cluster were segmented by the older detail prompt (about one sentence per detail); the supplement's segments come from the
newer run (about 3 sentences per detail). Within-story spectra (B, C) depend on the number of units, so the reported `kn` (k / (units - 1)), the size-adjusted
trend tests and `story_profiles.csv` (`n_units`) must be read with that in mind; panel A (whole-story vectors) does not depend on segmentation.
