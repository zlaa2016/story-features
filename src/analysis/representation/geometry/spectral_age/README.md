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

## Extracting vectors for a new model (Qwen) on the batch videos only
The cluster has no Qwen vectors in the layout `run.py` reads. `submit_extraction.sh` cuts the cohort to the batch videos
(`filter_cohort.py`, original stories only, missing videos listed in `spectral_age_cohort.missing_videos.csv`) and submits two chained
array jobs: whole-story + contextual detail vectors (`extract_global.py`) and isolated detail vectors (`extract_detail_controls.py`).
Outputs: `<root>/qwen3_8b_spectral_global/` and `<root>/qwen3_8b_spectral_detail_controls/`, which `MODEL_KEY=qwen3_8b` reads by default.
```bash
MODEL_KEY=qwen3_8b bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
```
Qwen3-8B reads up to 32k tokens (`MAX_TOKENS=32768`, longer stories skipped and reported).
