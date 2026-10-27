# Gist structure and schema dimensionality across ages

**Question.** Stories written for older children are more complex to begin with. Is their
information nevertheless *more* reducible to a few unique gists (relative to how many details they
have), and does the shared "schema" of an age band have a different number of dimensions that is
nested with the neighbouring ages?

## What the first run changed

The first within-story run used variance thresholds (components needed for 20-95% of variance). Three
problems: in-sample reconstruction becomes perfect as k approaches n (stories have ~90 details in
2,880 dimensions); the 20% end sits at 1-2 components for nearly every story (a floor, so nothing can
differ); and the 90% end is mostly noise. The measures here are **held out** (a story's gist subspace
is fitted on some of its details and tested on the rest) and **null-referenced** (a column-permutation
null keeps every per-dimension variance, rogue dimensions and noise level and removes all covariance).
`calibrate.py` shows on synthetic stories with known structure which statistics recover what.

## What the calibration shows (synthetic stories, 30 per band x 5 bands per scenario)

Standardised age effect per year, adjusted for story size (M1); * = permutation p < 0.05. Synthetic
embeddings carry real-embedding artefacts (power-law per-dimension variance, two rogue dimensions,
a shared offset). "Legacy" are the variance-threshold statistics from the first run.

| Planted with age | `gist_count_rel` | `gist_peak` | `event_E5` | legacy `kn90` | legacy `alpha` | legacy `head3` |
|---|---|---|---|---|---|---|
| nothing (null) | 0 | 0 | 0 | 0 | 0 | 0 |
| fewer gists per detail (the hypothesis) | **-0.33\*** | ~0 / +0.19\* | +0.43\* | +0.07 | +0.30\* | +0.55\* |
| more gists per detail (reverse) | **+0.33\*** | -0.16\* | -0.33\* | -0.05 | -0.32\* | -0.25\* |
| noisier details, same gists and size | **~0** (-0.09, +0.02 fixed) | **-0.34\*** | -0.34\* | **+0.31\*** | **-0.34\*** | **-0.34\*** |
| longer, same ratio of gists to details | 0 | 0 | 0 | 0 | +0.06 | -0.03 |
| longer, same number of gists | 0 | 0 | -0.12\* | 0 | 0 | -0.09\* |

Without the size term (M0) a story that is only longer shows `gist_count_rel` -0.24\* ("older compress
more") and legacy `kn90` -0.31\*, so size control is not optional.

Reading it: the legacy signature of the first real run (older: `kn90` up, `alpha` down, `head3` down)
is what *noisier, more idiosyncratic details* produce (row 4) and is also what "more gists per detail"
produces (row 3, except `kn90`). The new measures separate them: more gists per detail raises
`gist_count_rel`; idiosyncratic noise lowers `gist_peak` and `event_E5` and leaves `gist_count_rel`
near zero.

## Steps

| Step | Script | What it does |
|---|---|---|
| calibrate | `calibrate.py` | synthetic stories with planted gist ratio / size / noise; which statistics respond to which |
| select | `select_stories.py` | popular, representative, topic-matched, equal-sized age bands; balance report |
| within | `within_story.py` | per-story held-out gist curve, event structure, held-out VQ, duplicate-removed versions; age effects under four nested control models |
| schema | `schema_age.py` | per-band schema dimensionality, cross-age transfer and nestedness, per-story schema share |

### What the master table looks like (checked on `data/processed/master_stories.csv.gz`)
2,138 stories, 1,218 author groups (a few authors have 27-47 stories: Walliams, Willems, Seuss, Disney),
views for 100% and review counts for 99%. At overlap >= 0.5 the two-year bands 2-3 / 4-5 / 6-7 / 8-9 /
10-12 get 339 / 453 / 218 / 35 / 90 stories, and 956 stories have ranges mostly outside every band, so
the default bands are **2-3, 4-5, 6-7, 8-12**. Author clustering is large enough that the one-story-
per-author cap and the author-cluster bootstrap are not optional. `SELECT_MODE=all` keeps the
wide-range stories (banded by range midpoint, authors capped at 3, `range_width` as a covariate) as a
sensitivity analysis.

### Story selection (`select`)
Each story goes to the band containing most of its stated age range (overlap >= 0.5; ranges like 2-12
are left out). Stories are clustered by the mean of their detail vectors (topic clusters) and every
band gets the same topic mix. Inside each quota, greedy weighted facility location chooses stories that
cover the band's topic space, weighted by popularity (mean percentile of log views, log reviews, log
likes), with one story per author group and a channel cap. `--objective random` and `--topic-quota none`
are the control selections.

### Within-story measures (`within`)
* `gist_peak`: strength, how much of left-out details the best m components reconstruct beyond the null.
* `gist_count` / `gist_count_rel`: the m where that curve saturates (90% of peak), the estimated number
  of gists, and per detail. Only defined when the peak is meaningful (>= 0.02).
* `event_*`: variance explained by the best k *contiguous* segments minus a shuffled-order null.
* `vq_*`: the same with held-out k-means centroids (nonlinear "unique gists").
* `dd_*`: all of the above after near-duplicate details (refrains) are removed;
  `dup_share` and `mean_pairwise_cos` describe redundancy.

### Schema dimensionality (`schema`)
Details are expressed as deviations from their own story's mean (theme removed) and pooled by band
(equal stories, equal details per story, a label-free PCA to `--reduce-dim`).
* `D50 / D80 / D90`: components needed for that share of the plateau of the held-out curve (stories
  held out), minus a column-permutation null; `pa_dim`: components above a parallel-analysis threshold;
  bootstrap over stories.
* transfer: how well band a's subspace reconstructs band b's stories relative to b's own held-out
  fidelity, at m >= the larger band's dimension. `asymmetry_mean` < 0 means older subspaces explain
  younger stories better than the reverse (younger schema nested in older).
* subspace overlap and its decay with age distance; per-story `schema_share` (band schema vs the
  story's own gist subspace).
Every trend statistic gets a permutation null from shuffling band labels within topic x size strata.

## PC overlap between ages (`pc_overlap_by_age.py`)

The most direct view of age against the spectrum: for each age 2-12 (every flat-list story whose stated
range covers it), PCA of the story embeddings; the top components that capture 50 / 80 / 95% of the
variance; and the overlap `||U_a U_b^T||^2 / k` between ages. Plotted against age: the number of components,
the overlap with the next age, and the overlap with age 2. All cells are cut to the same size and every
number comes from random half-splits, so a same-age ceiling (two halves of one age), a permuted-age band
(age ranges shuffled among stories) and the chance level k/d are directly comparable to the observed curve.
`--source distilbert` runs on the local story vectors; `--source gpt_oss_global` on the cluster's
whole-story GPT-OSS vectors (`run_pc_overlap.slurm`). This is between-story structure, not within-story gist.

## Hypotheses and what would falsify them

* **H1 (within story).** Older stories need fewer gists *relative to their own size*: negative age
  effect on `gist_count_rel`, with strength (`gist_peak`) not lower. Primary test: `all_units`, model
  M3 (size, length, lexical, duplicates, topic, popularity held fixed). *Falsified* if the effect is
  zero or positive. Strength and count are separate: noise lowers strength without changing the count.
  `fixed_40` asks something different: gists per **equal-size sample** of details, so a story with
  more distinct events scores higher even if its proportion is unchanged (calibration
  `complexity_only`). Read the two together: negative in `all_units` and positive in `fixed_40`
  means "more total content, proportionally more reducible"; both negative means fewer gists at
  every scale.
* **H2 (schema).** Held-out schema dimensionality (`D80`, `pa_dim`) rises with age band and younger
  schemas are nested in older ones (`asymmetry_mean` < 0, overlap falling with age distance).
  *Falsified* by flat dimensionality, symmetric transfer, or trends inside the permutation null.

## Confounds and their controls

| Confound | Control |
|---|---|
| Story length / number of details | story size in every model from M1 up (without it a longer story looks "more compressible" by arithmetic: calibration `size_only`, M0); `--fixed-units` as a complement (see below) |
| Lexical complexity, detail length | M2 covariates: mean detail tokens, type-token ratio (300 words), word length, duplicate share, age-range width |
| Repetition and refrains | `dd_*` outcomes; `dup_share` covariate; schema `--dedupe` robustness run |
| Embedding artefacts (anisotropy, rogue dimensions, noise, n << d) | held-out curves; column-permutation null; calibration with those artefacts built in |
| Topic differences between ages | topic-matched selection; topic strata in permutations; topic dummies in M3 |
| Popularity, channel, author, series | popularity weights and covariate; channel cap; author cap; bootstrap resamples authors |
| Selection itself | `--objective random`, `--topic-quota none` variants; balance report |
| Noise vs gist count | `gist_peak` (strength) reported separately from `gist_count`; calibration `noise_only` |
| Wide or ambiguous age ranges | band by overlap >= 0.5, wide ranges excluded; `SELECT_MODE=all` sensitivity keeps them with `range_width` (M2) as a covariate |
| Attrition by length (short = young) | `inclusion_by_band.csv` in `within` and `schema` |
| One embedding model | rerun with `MODEL_KEY=qwen3_8b` (own `OUTPUT_ROOT`) and compare signs |
| Many outcomes | primary outcome fixed in advance; Benjamini-Hochberg q within each model |

## Run

```bash
git fetch origin && git checkout compress && git pull --ff-only
SUBSET_CSV=data/processed/flat_videos_D_2-12.csv STEPS=calibrate,select,within,schema \
sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/gist_structure/run.slurm
```

Variants, each with its own output folder:

```bash
# control selections
OBJECTIVE=random OUTPUT_ROOT=results/analysis/representation/geometry/gist_structure/gpt_oss_20b_random ...
TOPIC_QUOTA=none OUTPUT_ROOT=results/analysis/representation/geometry/gist_structure/gpt_oss_20b_notopic ...
# another model
MODEL_KEY=qwen3_8b ...
```

Check `selection/selection_summary.json` (selected per band, imbalanced variables), `balance_report.csv`
and both `inclusion_by_band.csv` files before reading any effect.

## Outputs (`<OUTPUT_ROOT>/`)
* `calibration/calibration_summary.csv`
* `selection/`: `selected_stories.csv`, `candidate_stories.csv`, `balance_report.csv`, `selection_summary.json`
* `within_story/{all_units,fixed_40}/`: `story_gist_profiles.csv`, `age_effects.csv` (outcome x model: standardised
  effect per year, author-cluster bootstrap interval, permutation p, BH q), `band_summary.csv`,
  `hypothesis_check.json`, `inclusion_by_band.csv`, `fig_gist_curves_by_band.png`, `fig_age_effects_by_model.png`
* `schema/`: `schema_dimensions.csv`, `schema_dimension_bootstrap.csv`, `schema_band_curves.csv`, `schema_transfer.csv`,
  `schema_trend_tests.json`, `story_schema_share.csv`, `schema_story_age_effects.csv`, `fig_schema_dimensionality.png`

## Reading the results
0. Check `split_half_reliability.csv` and `real_data_validation.json` in `within_story/<label>/`: a measure
   whose two halves of a story disagree cannot carry an age effect, and the merge control should give a
   merged pair's gist count of about the sum of its two stories (ratio near 1).
1. If the balance report or inclusion shows a band that differs in length, topic mix or attrition, say so first.
2. Compare the age effect across M0 -> M3. Shrinking toward zero means the raw effect was size, complexity or topic.
3. The effect must also appear in `fixed_40` and with `dd_` outcomes.
4. A trend in `schema` counts only if its permutation interval excludes the observed value.
5. Age and complexity are not separable in a natural corpus. M2/M3 show how much of an age effect survives
   once measured complexity, topic and popularity are held fixed; what remains is "age beyond those".
