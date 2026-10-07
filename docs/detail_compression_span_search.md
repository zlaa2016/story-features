# Per-detail compression by span search

For each summary detail, find the best consecutive run of source details that matches
its meaning. The size of that run is the compression of that detail. Run on stories that
have younger, actual and older summaries, flat across source age.

| Step | Code | Slurm |
|---|---|---|
| Pick the subset | `src/features/summaries/dataset_index/select_flat_subset.py` | `hpc/features/summaries/dataset_index/select_flat_subset.slurm` |
| Natural-language search (mpnet) | `src/analysis/natural_language/detail_compression/span_search.py` | `hpc/analysis/natural_language/detail_compression/span_search.slurm` |
| Representation search (GPT-OSS contextual + isolated) | `src/analysis/representation/detail_compression/span_search.py` | `hpc/analysis/representation/detail_compression/span_search.slurm` |
| Shared search core | `src/shared/span_search.py` | |

Submit everything (all stories that have all three summaries, no age flattening):
`NO_FLATTEN=1 SBATCH_ARGS="--account=<acct>" bash hpc/analysis/submit_span_search.sh`
Leave `NO_FLATTEN` unset to sample flat across source age instead.

Outputs from before the reorganisation are copied into the new `results/` layout with
`bash hpc/copy_old_outputs.sh`.

Results notebook: `notebooks/analysis/representation/detail_compression/span_search_by_age.ipynb`
(compression versus source age for all three methods and all three summaries).

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
