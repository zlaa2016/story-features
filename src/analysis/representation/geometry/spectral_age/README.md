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
The unit is the **story** (`story_key`, e.g. `abc123__story02`); a story inherits its video's stated range and view count. One rule for every batch:
the stated range must be under 8 (`upper - lower <= 7`). `data/processed/spectral_age/baseline_stories_span_lt8.csv` is that pool: 11,828 stories, all with
cleaned text and detail segments (9,353 have at least 20 details, the minimum for the within-story panels B and C; the others still count in panel A). A story counts at
every age its range covers. Stories covering each age: 4,585 / 7,580 / 8,814 / 8,741 / 7,155 / 4,936 / 3,296 / 1,135 / 789 / 464 / 404 for ages 2-12.

| Batch folder | Design | Cell size | Distinct stories |
|---|---|---|---|
| `batch_top800_ages2-8` | 800 most-viewed stories per age, ages 2-8 | 800 at every age | 2,113 |
| `batch_top400_ages2-12` | 400 most-viewed stories per age, ages 2-12 | 400 at every age | 1,740 |
| `batch_flat_ages2-8_span_lt8` | flat design: equal coverage at every age, ages 2-8 | 2,553 at every age | 5,287 |

`batch_flat_*` is a different design: the largest set of stories whose coverage is *exactly equal* at every age (a story counts at every age its range covers),
solved as an integer program (`subsets.py flat`, `baseline_stories_span_lt8.csv`); a cell is every selected story covering the age. As many stories as possible are kept at that coverage (5,287 of the 11,633 stories touching ages 2-8; age 8 has only 3,296 and the ranges force the rest out); among those,
stories of videos already queued for extraction are preferred, then the most viewed. Its stories' videos outside the first extraction list are in `extraction_videos_flat_extra.csv`; the 400 batch needs 25 more videos (`extraction_videos_top400_extra.csv`) and the final flat set 286 more (`extraction_videos_flat_extra2.csv`).
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

The 400 batch's 25 extra videos (27 stories; shard ids from 400):
```bash
D=data/processed/spectral_age; R=results/features/representation_extraction/causal_lm/detail_extraction
MODEL_KEY=gpt_oss_20b VIDEOS_CSV=$D/extraction_videos_top400_extra.csv SUPPLEMENT=$D/supplement_cohort_top400_extra.jsonl.gz COHORT_NAME=spectral_age_cohort_top400_extra_gpt_oss.jsonl SHARD_OFFSET=400 EXCLUDE_DIRS="$R/gpt_oss_20b_global_current" bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
```

The final flat set's 286 extra videos (shard ids from 500):
```bash
D=data/processed/spectral_age; R=results/features/representation_extraction/causal_lm/detail_extraction
MODEL_KEY=gpt_oss_20b VIDEOS_CSV=$D/extraction_videos_flat_extra2.csv SUPPLEMENT=$D/supplement_cohort_flat_extra2.jsonl.gz COHORT_NAME=spectral_age_cohort_flat_extra2_gpt_oss.jsonl SHARD_OFFSET=500 EXCLUDE_DIRS="$R/gpt_oss_20b_global_current" bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
```

## Curve-level age test (`age_tests.py`)
One test per panel on the whole "PCs needed vs share of variance" curve: T = mean over variance levels 20-95% of the age association of k(v).
B (within story): size-adjusted Spearman rho of k(v) / (units - 1) with the story's age midpoint, permuting ages within strata of story length (2,000 draws).
A (pooled): Spearman rho across ages of k(v), every age at the smallest cell size, permuted-age null (200 draws). T > 0: older ages need more PCs. Stories and ages are
treated as independent and no correction is applied (one test per panel). Written to `<run>/within_<view>/age_curve_test.json` and `<run>/pooled/age_curve_test.json`,
with `fig_age_curve_test.png` showing the association at each level. On a finished run the within-story test alone can be rerun without vectors:
```bash
PYTHONPATH=src python -m analysis.representation.geometry.spectral_age.age_tests --profiles results/analysis/representation/geometry/spectral_age/gpt_oss_20b/batch_flat_ages2-8_span_lt8_isolated/within_isolated/story_profiles.csv
```

## Random subset and the four panels

`batch_random100_ages2-12` draws up to 100 random stories per age (2-12) from the stories with vectors (stated range under 8); an age with fewer keeps all of them and the real counts are in
`<output>/selection/summary.json`.  Run with `STEPS=pooled,within,halves`; `fig_four_panels.png` in the output folder shows:

1. each story one vector, PCA over the stories of an age: PCs needed per share of variance (A);
2. each detail one vector, PCA within each story: PCs needed, averaged over the stories at each age (B);
3. whole-story PCs within an age: PC overlap between two random halves of the stories at that age (A's half-split, the diagonal of the age-by-age overlap);
4. details within a story: each story's details are split at random into two halves, each half gets its own PCA, and the overlap of the two sets of PCs is averaged over the stories at each age
   (`halves_<view>/fig_halves_overlap.png`, `overlap_halves.csv`; null = a half of another story of that age).  Details are never pooled across stories.

### Top-5 PC age test

Next to the curve-level tests each step writes one test on the first 5 PCs only, `age_top5_test.json` (rho, two-sided permutation p, null mean/sd):

* `pooled/`: Spearman rho across ages between age and the share of variance in the first 5 PCs of the pooled whole-story vectors (every age at the smallest cell size; permuted-age null).
* `within_<view>/`: size-adjusted Spearman rho between a story's age midpoint and the share of its own variance in its first 5 detail PCs (story length ranked out, ages shuffled inside strata of story length).
* `halves_<view>/`: the same for the overlap of the first 5 PCs of two halves of a story's details; `overlap_halves_top5.csv` has the per-age means and the other-story null.

One test per panel, no multiple-comparison correction; stories at several ages are used once per cell, so read neighbouring ages as non-independent.

### Does position in the story matter? (step `chunks`)

`STEPS=chunks` (details only) cuts each story's details, in story order, into up to 5 consecutive chunks (fewer for short stories: chunks of at least 7 details, never under 2 chunks) and, separately, into random chunks of the same sizes (10 draws per story).  Each chunk gets its own PCA.
For each design it records the overlap of the chunks' first 3 PCs and the cosine of their story-centred means.  The two designs are reported **separately and not combined**: ordered chunks by how far apart they sit in the story (1 to 4 chunks apart, and all pairs), random chunks over all pairs (they have no position).
Outputs in `chunks_<view>/`: `fig_chunks_ordered.png`, `fig_chunks_random.png`, `story_chunks.csv` (per story), `chunks_by_age.csv`, `chunk_age_tests_ordered.json`, `chunk_age_tests_random.json`.
Tests, run on age for each design on its own: each story's value against its age midpoint (size-adjusted Spearman, story length ranked out, ages shuffled inside strata of story length).
Chunk counts differ between short and long stories (`stories_by_n_chunks` in the JSON), which affects the level of the overlap, so read age trends within a design.
