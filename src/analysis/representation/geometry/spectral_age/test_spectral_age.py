from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.spectral_age import run, subsets


def make_baseline(rng: np.random.Generator, count: int = 90) -> pd.DataFrame:
    lower = rng.integers(2, 7, count)
    upper = lower + rng.integers(0, 3, count)
    views = rng.integers(10, 10_000, count)
    video = [f"v{i // 2:03d}" if i % 5 == 0 else f"v{i:03d}" for i in range(count)]  # some videos hold two stories
    ordinal = pd.Series(video).groupby(video).cumcount() + 1
    story_key = [f"{v}__story{n:02d}" for v, n in zip(video, ordinal)]
    return pd.DataFrame({"story_key": story_key, "story_id": [f"{i:06d}" for i in range(count)], "video_id": video, "story_title": "t", "range_lower": lower,
                         "range_upper": upper, "range_text": "x", "view_count": views, "log_views": np.log1p(views), "n_substories": 1, "channel": "c"})


class SubsetTests(unittest.TestCase):
    def test_top_viewed_covering_each_age_and_equal_cells(self) -> None:
        baseline = make_baseline(np.random.default_rng(0))
        membership, subset, summary = subsets.select_flat(baseline, [3, 4, 5], per_age=1000)
        self.assertEqual(summary["unit"], "story")
        sizes = membership.groupby("age").size()
        self.assertEqual(sizes.nunique(), 1)
        self.assertEqual(int(sizes.iloc[0]), min(v["candidates"] for v in summary["by_age"].values()))
        for age, group in membership.groupby("age"):
            covering = baseline[(baseline["range_lower"] <= age) & (baseline["range_upper"] >= age)]
            self.assertTrue(set(group["story_key"]) <= set(covering["story_key"]))
            expected = covering.sort_values(["log_views", "story_key"], ascending=[False, True]).head(len(group))
            self.assertEqual(set(group["story_key"]), set(expected["story_key"]))

    def test_shortfall_is_reported_and_available_filter_applies(self) -> None:
        baseline = make_baseline(np.random.default_rng(1))
        available = set(baseline["story_key"].iloc[:30])
        membership, _, summary = subsets.select_flat(baseline, [3, 4], per_age=400, available=available)
        self.assertTrue(set(membership["story_key"]) <= available)
        self.assertGreater(summary["by_age"][3]["short_of_target"], 0)
        self.assertTrue(summary["restricted_to_available_vectors"])


class MeasureTests(unittest.TestCase):
    def test_twonn_recovers_low_dimension(self) -> None:
        rng = np.random.default_rng(2)
        low = rng.normal(size=(600, 3)) @ rng.normal(size=(3, 40))
        high = rng.normal(size=(600, 40))
        self.assertLess(run.twonn(low), run.twonn(high))
        self.assertAlmostEqual(run.twonn(low), 3.0, delta=1.0)

    def test_spectrum_row_counts_components(self) -> None:
        rng = np.random.default_rng(3)
        rows = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 50))
        row = run.spectrum_row(rows / np.linalg.norm(rows, axis=1, keepdims=True))
        self.assertLessEqual(row["k_95"], 5)
        self.assertGreater(row["k_99"], 0)


def write_shards(root: Path, rng: np.random.Generator, baseline: pd.DataFrame, d: int = 32) -> tuple[Path, Path]:
    global_dir, detail_dir = root / "global", root / "isolated"
    global_dir.mkdir()
    detail_dir.mkdir()
    layers = np.array([0, 23], dtype=np.int16)
    story_vectors, story_rows, detail_vectors, detail_rows = [], [], [], []
    for number, row in baseline.iterrows():
        age = (row["range_lower"] + row["range_upper"]) / 2
        rank = int(2 + age)
        n = int(rng.integers(25, 40))
        details = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d)) + 0.05 * rng.normal(size=(n, d))
        story_vectors.append(details.mean(axis=0))
        story_rows.append({"record_id": f"{row['story_key']}::original", "record_type": "original", "video_id": row["video_id"], "story_key": row["story_key"]})
        detail_vectors.append(details)
        detail_rows += [{"record_id": f"{row['story_key']}::original", "record_type": "original", "story_key": row["story_key"], "video_id": row["video_id"],
                         "source_age": float(age), "detail_index": i} for i in range(n)]
    whole = np.stack(story_vectors).astype(np.float16)
    np.savez(global_dir / "global_features_000.npz", mean_vectors=np.stack([whole, whole * 0.5], axis=1), layers=layers)
    pd.DataFrame(story_rows).to_csv(global_dir / "global_metadata_000.csv", index=False)
    stacked = np.concatenate(detail_vectors).astype(np.float16)
    np.savez(detail_dir / "isolated_detail_features_000.npz", mean_vectors=np.stack([stacked, stacked * 0.5], axis=1), layers=layers)
    pd.DataFrame(detail_rows).to_csv(detail_dir / "isolated_detail_metadata_000.csv", index=False)
    return global_dir, detail_dir


class MultiFolderTests(unittest.TestCase):
    def test_folders_are_combined_and_first_folder_wins(self) -> None:
        rng = np.random.default_rng(5)
        full = make_baseline(rng, 40)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a", Path(tmp) / "b"
            first.mkdir()
            second.mkdir()
            global_a, detail_a = write_shards(first, rng, full.iloc[:25])
            global_b, detail_b = write_shards(second, rng, full.iloc[15:])  # videos 15-24 are in both
            ids, vectors = run.whole_story([global_a, global_b, Path(tmp) / "missing"], "mean_vectors", -1)
            self.assertEqual(sorted(ids.tolist()), sorted(full["story_key"].tolist()))
            self.assertEqual(len(vectors), 40)
            details = run.detail_table([detail_a, detail_b], "isolated")
            overlapped = details[details["story_key"].isin(full["story_key"].iloc[15:25])]
            self.assertEqual(set(overlapped["folder"]), {str(detail_a)})
            loaded = run.detail_vectors(details, "isolated", -1, None)
            self.assertEqual(loaded.shape[0], len(details))
            self.assertTrue(np.isfinite(loaded).all() and np.abs(loaded).sum(axis=1).min() > 0)


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_end_to_end_on_synthetic_shards(self) -> None:
        rng = np.random.default_rng(4)
        baseline = make_baseline(rng, 120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir, detail_dir = write_shards(root, rng, baseline)
            baseline.to_csv(root / "baseline.csv", index=False)
            batch = root / "batch"
            membership, subset, summary = subsets.select_flat(baseline, [3, 4, 5, 6], per_age=25)
            subsets.write_batch(batch, membership, subset, summary)
            out = root / "out"
            run.main(["--batch-dir", str(batch), "--baseline-csv", str(root / "baseline.csv"), "--global-dir", str(global_dir), "--detail-dir", str(detail_dir),
                      "--reselect-available", "--standardize", "--splits", "3", "--permutations", "3", "--min-units", "20", "--reduce-dim", "16",
                      "--trend-permutations", "20", "--output-dir", str(out)])
            for name in ("fig_spectrum_by_age.png", "selection/membership.csv", "pooled/fig_age_by_variance.png", "pooled/other_reductions_by_age.csv",
                         "within_isolated/story_profiles.csv", "within_isolated/k_by_age_variance_within.csv", "within_isolated/age_trend_within.csv",
                         "within_isolated/fig_within_and_pooled_within.png", "within_isolated/k_by_age_variance_within_pooled.csv"):
                self.assertTrue((out / name).exists(), name)
            k = pd.read_csv(out / "within_isolated/k_by_age_variance_within.csv")
            at_90 = k[np.isclose(k["variance"], 0.9)].set_index("age")["mean_k"]
            self.assertGreater(at_90.loc[6], at_90.loc[3])
            self.assertEqual(json.loads((out / "selection/summary.json").read_text())["cell_size"], 25)


if __name__ == "__main__":
    unittest.main()
