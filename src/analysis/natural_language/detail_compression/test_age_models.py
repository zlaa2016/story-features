import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels.formula.api")
models = importlib.import_module("analysis.natural_language.detail_compression.age_models")


def synthetic(n_stories=120, slope=0.12, seed=0):
    rng = np.random.default_rng(seed)
    s_rows, d_rows = [], []
    for k in range(n_stories):
        age = rng.uniform(5, 10)
        story_effect = rng.normal(0, 0.2)
        for ci, cond in enumerate(["younger", "actual", "older"]):
            n_src, n_sum = int(rng.integers(40, 120)), int(rng.integers(8, 20))
            record = f"s{k}::{cond}"
            spans = np.clip(np.round(2 ** (0.6 + slope * (age - 7) + story_effect + rng.normal(0, 0.4, n_sum))), 1, 10)
            d_rows.append(pd.DataFrame({"story_key": f"s{k}", "condition": cond, "record_id": record, "source_age": age, "length_penalty": 0.0, "span_len": spans}))
            s_rows.append({"story_key": f"s{k}", "condition": cond, "record_id": record, "source_age": age, "length_penalty": 0.0,
                           "n_source_details": n_src, "n_summary_details": n_sum, "mean_log2_word_ratio": rng.normal(1, .3),
                           "mean_score": 2.4 + rng.normal(0, .2), "mean_null_score": 1.8 + rng.normal(0, .2)})
    return pd.concat(d_rows, ignore_index=True), pd.DataFrame(s_rows)


def test_planted_age_slope_is_recovered_and_tests_run():
    details, summaries = synthetic()
    d, s = models.prepare(details, summaries)
    slopes, lr, fixed, icc = models.mixed_models(s, None)
    pooled = slopes[(slopes.outcome == "mean_log2_span") & (slopes.model == "summary level")]
    assert (pooled.slope_per_year > 0.04).all() and (pooled.slope_per_year < 0.25).all()
    assert (pooled.p < 0.05).all()
    assert lr[lr.test.str.startswith("age effect")].p.iloc[0] < 0.01
    rank = models.rank_tests(s, n_boot=100, n_perm=200)
    assert rank[rank.group == "all summary types, story mean"].statistic.iloc[0] > 0
