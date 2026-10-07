import numpy as np
import pandas as pd

from shared.retention_shift import adjust_shift, kernel_smooth, smooth_with_band, story_measures, windowed_odds


def synthetic(n_stories=80, seed=0, strength=lambda age: 1.5 * (age - 4) / 4):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stories):
        age = rng.uniform(2, 8)
        n = 40
        shift = rng.normal(0.3, 0.1, n)
        logit = -1.0 - strength(age) * (shift - 0.3) / 0.1
        keep = rng.random(n) < 1 / (1 + np.exp(-logit))
        rows.append(pd.DataFrame({"story_key": f"s{s}", "source_age": age, "shift": shift, "y": keep.astype(float),
                                  "position_norm": np.linspace(0, 1, n), "detail_len": rng.integers(3, 30, n)}))
    return pd.concat(rows, ignore_index=True)


def test_auc_is_above_half_when_dropped_details_have_larger_shift_and_grows_with_age():
    frame = synthetic()
    table = story_measures(frame)
    assert table["auc"].mean() > 0.5
    old, young = table[table.source_age > 6], table[table.source_age < 3.5]
    assert old["auc"].mean() > young["auc"].mean()
    curve = smooth_with_band(table, "auc", np.array([2.5, 7.0]), bandwidth=0.8, n_boot=60)
    assert curve["estimate"].iloc[1] > curve["estimate"].iloc[0]


def test_no_relation_gives_chance_level():
    frame = synthetic(strength=lambda age: 0.0)
    assert abs(story_measures(frame)["auc"].mean() - 0.5) < 0.03


def test_adjust_shift_removes_a_pure_position_trend():
    rng = np.random.default_rng(1)
    pos = np.tile(np.linspace(0, 1, 50), 20)
    frame = pd.DataFrame({"position_norm": pos, "detail_len": rng.integers(3, 30, len(pos)), "shift": 0.2 + 0.3 * pos})
    assert adjust_shift(frame).abs().max() < 0.03


def test_kernel_smooth_recovers_a_line():
    x = np.linspace(2, 8, 200)
    estimate = kernel_smooth(x, 2 * x, np.ones_like(x), np.array([5.0]), 0.5)
    assert abs(estimate[0] - 10.0) < 0.05


def test_windowed_odds_below_one_when_shift_is_dropped():
    frame = synthetic(n_stories=60)
    frame["centrality_pct"] = np.random.default_rng(2).random(len(frame))
    frame["log_len"] = np.log(frame["detail_len"])
    curve = windowed_odds(frame, np.array([5.0, 7.0]), n_boot=10)
    assert (curve["estimate"] < 1).all() and curve["estimate"].iloc[1] < curve["estimate"].iloc[0]
