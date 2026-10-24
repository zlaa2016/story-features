# Within-story compression profile by age

Each story's own detail vectors (units x dimensions) are analysed separately, so no PCA basis is
shared across stories. Per story: variance spectrum (`k(v)` for v = 0.20..0.90, head share,
decay exponent `alpha`, participation ratio), incremental dimensionality `PR(t)` against a
shuffled-order null, and position drift (variance explained by a smooth trend along the story,
spectrum after detrending). Spectrum statistics also get an excess over random units of other
stories with the same number of units.

Stories are assigned one age each (`--per-age` stories per age, only ages inside the range a story
covers, narrow-range stories preferred), aggregated by age with story-level bootstrap intervals,
and tested with the size-adjusted (partial) Spearman rho against age. Its permutation null shuffles ages within strata of number of units and recomputes the same statistic (`p_partial`); `partial_ci_*` are story-bootstrap intervals.
`--n-assignments` repeats the random assignment; `age_tests_across_assignments.csv` gives the median.

Views: `isolated` (each detail encoded alone; content structure) and `contextual` (detail states
inside the full story; also carries context accumulation, so read position drift with care).

Outputs per view (`<output-dir>/<view>/`): `story_profiles.csv`, `assignments.csv`,
`age_summary_rNN.csv`, `age_tests_rNN.csv`, `hypothesis_check_rNN.json`, `fig_*.png`.

## Reading the k(v) profile from 95% down to 20%

`k_age_profile_rNN.csv` and `fig_k_age_profile.png` give beta(v): the age effect on log k(v),
per year of age, adjusting for log n_units, at every v from 0.20 to 0.95 (raw k, and k relative
to the random-unit null). `fig_k_shape_by_age_group.png` shows mean k(v), k(v)/k(0.95) and
k(v)/null by age band.

* beta high at v = 0.95 and falling to about 0 at v = 0.20: older stories start with more
  components but converge at the low end (the head is similar).
* beta flat across v: a proportional shift, both drop at the same relative rate.
* beta growing as v falls: older stories keep a larger low end (the drop is slower).

`FIXED_UNITS=N` subsamples every story to exactly N units (shorter stories are dropped), which
removes "has more details" as an explanation for a higher start. Use a separate `OUTPUT_DIR`.

Pre-stated hypothesis (older stories reduce to a few gist components despite higher
dimensionality): partial rho (controlling log n_units) of `kn90` with age > 0 **and** of `alpha`
with age > 0. The rival, homogeneous young stories, gives `kn90` < 0 with `alpha` > 0 or
`head3` falling with age.

```bash
git fetch origin && git checkout compress && git pull --ff-only
SUBSET_CSV=data/processed/flat_videos_D_2-12.csv \
sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/within_story_compression/run.slurm
```

`SUBSET_CSV` needs columns `video_id, range_lower, range_upper`. Without it each story is placed
at its rounded `source_age` midpoint and the range-aware matching is not used.
Other variables: `MODEL_KEY`, `ISOLATED_DIR`, `CONTEXTUAL_DIR`, `VIEWS`, `LAYER`, `PER_AGE`,
`N_ASSIGNMENTS`, `EQUALIZE=1`, `OUTPUT_DIR`.
