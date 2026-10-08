# Per-detail compression by span search

For each summary detail, find the best consecutive run of source details that matches
its meaning. The size of that run is the compression of that detail. Run on stories that
have younger, actual and older summaries, flat across source age.

| Step | Code | Slurm |
|---|---|---|
| Pick the subset | `src/features/summaries/dataset_index/select_flat_subset.py` | `hpc/features/summaries/dataset_index/select_flat_subset.slurm` |
| Natural-language search (mpnet) | `src/analysis/natural_language/detail_compression/span_search.py` | `hpc/analysis/natural_language/detail_compression/span_search.slurm` |
| Representation search (GPT-OSS contextual + isolated) | `src/analysis/representation/detail_compression/span_search.py` | `hpc/analysis/representation/detail_compression/span_search.slurm` |
| Contextual-vs-isolated shift, all layers | `src/analysis/representation/detail_compression/contextualization_shift.py` | `hpc/analysis/representation/detail_compression/contextualization_shift.slurm` |
| Shared search core | `src/shared/span_search.py` | |

Submit everything (all stories that have all three summaries, no age flattening):
`NO_FLATTEN=1 SBATCH_ARGS="--account=<acct>" bash hpc/analysis/submit_span_search.sh`
Leave `NO_FLATTEN` unset to sample flat across source age instead.

Outputs from before the reorganisation are copied into the new `results/` layout with
`bash hpc/copy_old_outputs.sh`.

Results notebook: `notebooks/analysis/representation/detail_compression/span_search_by_age.ipynb`
(compression versus source age for all three methods and all three summaries) and
`contextual_vs_isolated_by_age.ipynb` (contextualization shift versus source and target age, by layer, linked to span length).

On the current cluster outputs, 138 stories have younger, actual and older summaries (3,056 stories have originals).

## Outputs

`results/features/summaries/dataset_index/flat_subset.csv` (+ `_by_age.csv`), and under
`results/analysis/{natural_language,representation}/detail_compression/span_search/`:
`details.csv` (one row per summary detail), `summaries.csv` (one row per summary),
`audit.csv`, `config.json`. Representation runs add `contextual_mean/` and `isolated_mean/`.

`details.csv` columns: `span_len` (source details, primary), `span_share` (span_len / story
details), `relative_compression` (span_len x summary details / story details; 1 = average),
`word_ratio` and `log2_word_ratio`, `token_ratio` (representation runs only), `score`,
`null_score` (same search against another story's source), `best_unconstrained_span_len`.

## Caveats

- Ratios depend on the length penalty, so the search is repeated for `LENGTH_PENALTIES`
  (default 0, 0.1, 0.2, in score-spread units). Report all three, not one.
- Consecutive spans over-widen details that blend distant events; compare against
  `best_unconstrained_span_len` and the existing non-consecutive control in `detail_alignment`.
- Isolated span vectors are token-weighted averages of isolated detail vectors, not joint encodings.
- `score - null_score` is the evidence a match beats chance; without it a ratio is not interpretable.

## Running the notebooks on the cluster

Both notebooks read `results/`. `contextual_vs_isolated_by_age.ipynb` also needs the mpnet alignment
output `results/analysis/natural_language/detail_alignment/original_detail_retention.csv` (copy it from
`mechanistic_interpretability/outputs/13_structured_detail_alignment` with `hpc/copy_old_outputs.sh`) for
its retained-versus-omitted sections; without it those sections are skipped.

Execute headless inside the container (see the job in the project notes) with
`jupyter nbconvert --to notebook --execute <notebook> --ExecutePreprocessor.kernel_name=story-features-direct`
and set `RESULTS_DIR` to the absolute path of `results/`.

## Statistics: does compression change with source age?

`src/analysis/natural_language/detail_compression/age_models.py` reads `details.csv` and `summaries.csv` at one
length penalty (default 0, none) and writes tables and a forest plot to `results/analysis/natural_language/detail_compression/stats/`.
It fits mixed models with a random intercept per story (the younger, actual and older summaries of one story are
repeated measurements), likelihood-ratio tests for the age effect, the age x summary-type interaction and a story-length
control, a detail-level model with summaries nested in stories, rank-based tests (Spearman with permutation p-values,
Friedman with Holm-corrected Wilcoxon follow-ups) and a check that matches beat the null search. Compression outcomes are
on a log2 scale, so a slope of s per year multiplies the span by 2**s per year.

    PYTHONPATH=src python src/analysis/natural_language/detail_compression/age_models.py
