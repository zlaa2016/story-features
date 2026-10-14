# Per-detail compression by span search

For each summary detail, find the best consecutive run of source details that matches
its meaning. The size of that run is the compression of that detail. Every story in the cohort with a summary in a confirmatory condition is used, with every summary it has. No story is
chosen, balanced across age, or required to have all three conditions; the mixed models handle the unbalanced design.

| Step | Code | Slurm |
|---|---|---|
| Natural-language search (mpnet) | `src/analysis/natural_language/detail_compression/span_search.py` | `hpc/analysis/natural_language/detail_compression/span_search.slurm` |
| Sentence-level counterpart (details per summary *sentence*, no vectors) | same script with `--summary-segments` | `hpc/analysis/natural_language/detail_compression/span_search_sentence.slurm` |
| Representation search (GPT-OSS contextual + isolated) | `src/analysis/representation/detail_compression/span_search.py` | `hpc/analysis/representation/detail_compression/span_search.slurm` |
| Contextual-vs-isolated shift, all layers | `src/analysis/representation/detail_compression/contextualization_shift.py` | `hpc/analysis/representation/detail_compression/contextualization_shift.slurm` |
| Shared search core | `src/shared/span_search.py` | |

Submit everything: `SBATCH_ARGS="--account=<acct>" bash hpc/analysis/submit_span_search.sh`

Outputs from before the reorganisation are copied into the new `results/` layout with
`bash hpc/copy_old_outputs.sh`.

Results notebook: `notebooks/analysis/representation/detail_compression/span_search_by_age.ipynb`
(compression versus source age for all three methods and all three summaries) and
`contextual_vs_isolated_by_age.ipynb` (contextualization shift versus source and target age, by layer, linked to span length).

On the current cluster outputs, 138 stories have younger, actual and older summaries (3,056 stories have originals).

## Outputs

Under `results/analysis/{natural_language,representation}/detail_compression/span_search/`:
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

## No subsetting

There is no subset-selection step. Selecting stories to be flat across age, or only stories with all three summaries, is a
choice that can shape the effect size, so every story and summary is analysed. The only exclusion is by unit: summaries
segmented at sentence level (`local_sentence`) are a separate measurement (source details per summary *sentence*, run by
`span_search_sentence.slurm` straight from the sentence file) and are skipped by the detail-level runs. The paired tests in
`age_models.py` (Friedman, Wilcoxon) need all three summaries, so they use only the stories that have them and report how many;
the mixed models use every story.
