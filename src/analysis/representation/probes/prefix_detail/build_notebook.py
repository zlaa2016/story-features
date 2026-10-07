#!/usr/bin/env python3
"""Build the GPT-OSS prefix/detail probe analysis notebook."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "notebooks/analysis/representation/probes/gpt_oss_prefix_detail_probe_analysis.ipynb"


def markdown(source: str, index: int) -> dict:
    return {
        "cell_type": "markdown", "id": f"md-{index:02d}", "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str, index: int) -> dict:
    return {
        "cell_type": "code", "id": f"code-{index:02d}", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


cells = [
    markdown(r"""
# GPT-OSS contextual detail probes

This notebook keeps four questions separate:

1. **Centrality retention:** are retained original details more central than omitted details, where
   centrality is the mean cosine with every other original detail in the same story?
2. **Contextualization:** how does the identical detail change between isolation and its native
   preceding context?
3. **Narrative order:** as context accumulates, does the true order differ from shuffled or foreign
   token-matched prefixes? Cubic splines model this nonlinear trajectory.
4. **Semantic compression span:** how many contiguous full-text details contribute to each summary
   detail? That analysis is produced by `summary_detail_window_index.ipynb`; its primary discrete
   index and spline-FWHM results are brought in here when available.

Actual-summary retention is used throughout. The two-row models contain exactly one retained and
one omitted mean per summary. Positive age slopes mean the displayed outcome increases with source
age.
""", 0),
    code(r"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path(os.environ.get('STORY_FEATURES_ROOT', Path.cwd())).resolve()
if not (ROOT / 'mechanistic_interpretability').exists():
    raise RuntimeError(f'Run from the story-features repository; got {ROOT}')

ANALYSIS_DIR = Path(os.environ.get(
    'PREFIX_PROBE_ANALYSIS_DIR',
    ROOT / 'output/prefix_detail_probes/gpt_oss_20b_native',
)).resolve()
SEMANTIC_DIR = Path(os.environ.get(
    'PM3_REPRESENTATION_CONTRAST_DIR',
    ROOT / 'results/features/summaries/runs/pm3_grounding_combined/representation_contrast',
)).resolve()

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 180)
print('Repository:', ROOT)
print('Probe analysis:', ANALYSIS_DIR)
print('Semantic window analysis:', SEMANTIC_DIR)

def load_csv(name, **kwargs):
    path = ANALYSIS_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()

metrics = load_csv('probe_metrics.csv.gz', low_memory=False)
two_rows = load_csv('actual_two_row_retention_means.csv')
tests = load_csv('actual_retention_hypothesis_tests.csv')
pooling = load_csv('mean_vs_last_pooling_equivalence.csv')
spline_tests = load_csv('prefix_spline_tests.csv')
spline_predictions = load_csv('prefix_spline_predictions.csv')

print(f'probe rows × layers: {len(metrics):,}')
print(f'stories: {metrics.story_key.nunique() if not metrics.empty else 0:,}')
""", 1),
    markdown(r"""
## RQ1. Are retained original details more central than omitted details?

For original detail (i) in a story with (N) details:

\[
C_i=\frac{1}{N-1}\sum_{j\ne i}\cos(z_i,z_j).
\]

Here (z_i) is the isolated, mean-pooled GPT-OSS representation at the final selected layer.
The table below has exactly two rows per actual summary: the mean (C_i) among retained details
and the mean among omitted details. The mixed model is

\[
C\sim\text{source age}*\text{retention}+(1|\text{story}).
\]
""", 2),
    code(r"""
centrality_rows = (two_rows[two_rows.metric.eq('isolated_semantic_centrality')].copy()
                   if 'metric' in two_rows else pd.DataFrame())
centrality_tests = (tests[tests.metric.eq('isolated_semantic_centrality')].copy()
                    if 'metric' in tests else pd.DataFrame())
display(centrality_rows.head(12))
display(centrality_tests)

if not centrality_rows.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'retained': '#4C78A8', 'omitted': '#E15759'}
    for status, block in centrality_rows.groupby('retention_status', observed=True):
        observed = block.groupby('source_age', as_index=False).value.mean()
        ax.plot(observed.source_age, observed.value, marker='o', label=status,
                color=colors[str(status)])
    ax.set(xlabel='Source age', ylabel='Mean isolated semantic centrality',
           title='Two rows per actual summary')
    ax.legend(title='Original detail')
    plt.tight_layout()
    plt.show()
""", 3),
    markdown(r"""
## RQ2. How does native preceding context change the same detail?

The target token IDs are identical. `first`, `last`, and `mean` differ only in which target-token
hidden states are read. `detail_nll_delta_from_isolated < 0` means that native context makes the
detail more expected. The fixed readout measures the downstream change after the whole detail.
""", 4),
    code(r"""
if not metrics.empty:
    final_layer = metrics.layer.max()
    native = metrics[(metrics.layer == final_layer) & metrics.prefix_condition.eq('true') & metrics.native_prefix]
    display(native[[
        'mean_contextualization_shift', 'last_contextualization_shift',
        'detail_nll_delta_from_isolated', 'mean_token_js_from_isolated',
        'readout_hidden_shift', 'readout_output_js',
    ]].describe().T)
display(pooling)
""", 5),
    markdown(r"""
## RQ3. Does true narrative order beat shuffled or foreign context?

A cubic B-spline models each outcome over normalized accumulated-prefix position. Story-clustered
standard errors account for repeated details, boundaries, and permutations. The joint Wald test
asks whether the complete shuffled or foreign curve differs from the true-order curve.
""", 6),
    code(r"""
display(spline_tests)
if not spline_predictions.empty and 'metric' in spline_predictions:
    outcomes = list(spline_predictions.metric.drop_duplicates())
    colors = {'true': '#4C78A8', 'shuffled': '#F28E2B', 'foreign': '#9C755F'}
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5.4 * len(outcomes), 4.6), squeeze=False)
    for ax, outcome in zip(axes[0], outcomes):
        data = spline_predictions[spline_predictions.metric.eq(outcome)]
        for condition, block in data.groupby('prefix_condition', observed=True):
            color = colors[str(condition)]
            ax.plot(block.prefix_fraction, block.fitted, label=condition, color=color)
            ax.fill_between(block.prefix_fraction, block.ci95_low, block.ci95_high,
                            color=color, alpha=.14)
        ax.set(xlabel='Accumulated prefix boundary', ylabel=outcome.replace('_', ' '))
    axes[0, -1].legend(title='Prefix')
    plt.tight_layout()
    plt.show()
else:
    print('No all-prefix spline results in this run. Execute the prefix-mode=all control run.')
""", 7),
    markdown(r"""
## RQ4. How many full-text details does each summary detail compress?

This is the natural-language/semantic moving-window analysis, not the causal prefix experiment.
For every summary detail it searches contiguous original-detail windows. `best_window_len` is the
discrete summarizing index. `spline_fwhm` is the width of the similarity peak across original-detail
positions. Both are shown for TF-IDF, isolated MiniLM, and contextual MiniLM when those results exist.
""", 8),
    code(r"""
semantic_path = SEMANTIC_DIR / 'winidx_summary_details.csv.gz'
if semantic_path.exists():
    semantic = pd.read_csv(semantic_path, low_memory=False)
    summary = semantic.groupby('representation', observed=True).agg(
        summary_details=('best_window_len', 'size'),
        mean_summarizing_index=('best_window_len', 'mean'),
        median_summarizing_index=('best_window_len', 'median'),
        share_fusing_2plus=('best_window_len', lambda x: (x >= 2).mean()),
        mean_spline_fwhm=('spline_fwhm', 'mean'),
    )
    display(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for representation, block in semantic.groupby('representation', observed=True):
        axes[0].hist(block.best_window_len, bins=np.arange(.5, 13.5), density=True,
                     histtype='step', linewidth=1.7, label=representation)
        axes[1].hist(block.spline_fwhm.dropna(), bins=35, density=True,
                     histtype='step', linewidth=1.7, label=representation)
    axes[0].set(xlabel='Best contiguous window length', ylabel='Proportion',
                title='Discrete summarizing index')
    axes[1].set(xlabel='Similarity-curve spline FWHM', ylabel='Density',
                title='Continuous compression span')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
else:
    print('Semantic window output not found:', semantic_path)
    print('Run scripts/analyze_summary_window_index.py or open notebooks/summary_detail_window_index.ipynb.')
""", 9),
    markdown(r"""
## Reading the results

- Centrality is retained as a continuous average-cosine quantity; the central/peripheral split is
  only a secondary visualization.
- The semantic spline asks **where and how broadly a summary detail is grounded in the full text**.
- The GPT-OSS prefix spline asks **how a fixed detail changes as preceding context accumulates**.
- A true-versus-shuffled difference supports sensitivity to sequential order only after comparison
  with the token-length-matched foreign condition.
""", 10),
]

payload = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "story-features", "language": "python", "name": "story-features"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    pass
