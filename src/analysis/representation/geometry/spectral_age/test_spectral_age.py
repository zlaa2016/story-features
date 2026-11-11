from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.spectral_age import age_tests, run, subsets


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


class WriteCsvTests(unittest.TestCase):
    def test_falls_back_when_the_csv_module_rejects_a_field(self) -> None:
        import csv
        from unittest import mock

        from shared.io import write_csv

        real, calls = pd.DataFrame.to_csv, {"n": 0}

        def flaky(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise csv.Error("need to escape, but no escapechar set")
            return real(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pd.DataFrame, "to_csv", flaky):
            write_csv(Path(tmp) / "t.csv", pd.DataFrame({"a": ["x\x00y", "ok"], "b": [1, 2]}))
            back = pd.read_csv(Path(tmp) / "t.csv")
        self.assertEqual(back["a"].tolist(), ["xy", "ok"])


class RandomDesignTests(unittest.TestCase):
    def test_random_draw_is_per_age_seeded_restricted_and_keeps_small_ages_whole(self) -> None:
        baseline = make_baseline(np.random.default_rng(12), 120)
        first = subsets.select_random(baseline, [3, 4, 5, 6], 12, seed=1)
        again = subsets.select_random(baseline, [3, 4, 5, 6], 12, seed=1)
        other = subsets.select_random(baseline, [3, 4, 5, 6], 12, seed=2)
        self.assertTrue(first[0].equals(again[0]))
        self.assertFalse(first[0].equals(other[0]))
        membership, subset, summary = first
        self.assertEqual(summary["design"], "random")
        for age, found in summary["by_age"].items():
            self.assertEqual(found["selected"], min(12, found["candidates"]))
            covering = baseline[(baseline["range_lower"] <= age) & (baseline["range_upper"] >= age)]
            self.assertTrue(set(membership.loc[membership["age"] == age, "story_key"]) <= set(covering["story_key"]))
        available = set(baseline["story_key"].iloc[:30])
        self.assertTrue(set(subsets.select_random(baseline, [3, 4], 50, available=available)[0]["story_key"]) <= available)


class FlatDesignTests(unittest.TestCase):
    def test_flat_design_has_equal_coverage_at_every_age_and_is_maximal(self) -> None:
        baseline = make_baseline(np.random.default_rng(6), 150)
        ages = [3, 4, 5, 6]
        membership, subset, summary = subsets.solve_flat(baseline, ages)
        sizes = membership.groupby("age").size()
        self.assertEqual(sizes.nunique(), 1)
        self.assertEqual(int(sizes.iloc[0]), summary["cell_size"])
        covering = {a: ((subset["range_lower"] <= a) & (subset["range_upper"] >= a)).sum() for a in ages}
        self.assertEqual(set(covering.values()), {summary["cell_size"]})
        again = subsets.solve_flat(baseline, ages, target=summary["cell_size"] + 1)[2]
        self.assertEqual(again["cell_size"], summary["cell_size"])  # the cap above the optimum changes nothing

    def test_preferred_videos_are_chosen_first_and_availability_limits_the_pool(self) -> None:
        baseline = make_baseline(np.random.default_rng(7), 150)
        prefer = set(baseline["video_id"].iloc[:40])
        _, subset, _ = subsets.solve_flat(baseline, [3, 4, 5], prefer_videos=prefer)
        _, plain, _ = subsets.solve_flat(baseline, [3, 4, 5])
        self.assertGreaterEqual(subset["video_id"].isin(prefer).sum(), plain["video_id"].isin(prefer).sum())
        available = set(baseline["story_key"].iloc[:50])
        membership, _, summary = subsets.solve_flat(baseline, [3, 4, 5], available=available)
        self.assertTrue(set(membership["story_key"]) <= available)
        self.assertTrue(summary["restricted_to_available_vectors"])


def synthetic_profiles(rng: np.random.Generator, count: int, slope: float) -> pd.DataFrame:
    ages = rng.integers(2, 9, count).astype(float)
    units = rng.integers(25, 120, count).astype(float)
    frame = {"age_mid": ages, "n_units": units}
    for v in range(10, 100):
        base = 1 + (v / 100) * 0.35 * units * (1 + slope * (ages - 5))
        frame[f"k_{v}"] = np.maximum(1, np.round(base + rng.normal(0, 1.5, count)))
    return pd.DataFrame(frame)


class CurveTestTests(unittest.TestCase):
    def test_within_story_curve_test_detects_a_planted_age_effect_and_not_a_null(self) -> None:
        rng = np.random.default_rng(10)
        effect = age_tests.within_curve_test(synthetic_profiles(rng, 300, 0.08), permutations=300)
        self.assertGreater(effect["T"], 0.2)
        self.assertLess(effect["p_two_sided"], 0.01)
        null = age_tests.within_curve_test(synthetic_profiles(rng, 300, 0.0), permutations=300)
        self.assertGreater(null["p_two_sided"], 0.05)
        self.assertEqual(effect["levels_used"], len([v for v in effect["curve"].values() if v is not None]))

    def test_pooled_curve_test_detects_dimensions_growing_with_age(self) -> None:
        from analysis.representation.geometry.gist_structure import pc_overlap_by_age as overlap

        rng = np.random.default_rng(11)
        count, dim = 900, 40
        lo = rng.integers(2, 8, count)
        hi = lo + rng.integers(0, 2, count)
        age = (lo + hi) / 2
        scores = np.stack([rng.normal(size=count) * (rng.random(count) < (0.2 + 0.1 * (age - 2) / 6 * k)) for k in range(1, dim + 1)], axis=1)
        x = scores @ rng.normal(size=(dim, dim)) + 0.05 * rng.normal(size=(count, dim))
        overlap.AGES = np.arange(2, 9)
        result = age_tests.pooled_curve_test(x, lo, hi, rng.random(count), np.arange(2, 9), 60, overlap.multi_cells, permutations=30)
        self.assertEqual(result["panel"], "A pooled")
        self.assertTrue(np.isfinite(result["T"]))
        self.assertGreaterEqual(result["p_two_sided"], 1 / 31)


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


class FlatPipelineTests(unittest.TestCase):
    def test_flat_batch_is_reselected_on_available_stories_and_runs(self) -> None:
        rng = np.random.default_rng(8)
        baseline = make_baseline(rng, 120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir, detail_dir = write_shards(root, rng, baseline.iloc[:90])  # vectors exist for 90 of 120 stories
            baseline.to_csv(root / "base.csv", index=False)
            membership, subset, summary = subsets.solve_flat(baseline, [3, 4, 5, 6], baseline_name="base.csv")
            subsets.write_batch(root / "batch", membership, subset, summary)
            out = root / "out"
            run.main(["--batch-dir", str(root / "batch"), "--baseline-csv", str(root / "base.csv"), "--global-dir", str(global_dir), "--detail-dir", str(detail_dir),
                      "--reselect-available", "--standardize", "--splits", "2", "--permutations", "2", "--reduce-dim", "16", "--trend-permutations", "10", "--output-dir", str(out)])
            chosen = json.loads((out / "selection/summary.json").read_text())
            self.assertEqual(chosen["design"], "flat")
            self.assertLess(chosen["cell_size"], summary["cell_size"])
            self.assertTrue(set(pd.read_csv(out / "selection/membership.csv")["story_key"]) <= set(baseline["story_key"].iloc[:90]))
            self.assertTrue((out / "fig_spectrum_by_age.png").exists())
            self.assertTrue((out / "within_isolated/age_curve_test.json").exists())


class RandomPipelineTests(unittest.TestCase):
    def test_random_batch_runs_pooled_within_and_halves_overlap(self) -> None:
        rng = np.random.default_rng(13)
        baseline = make_baseline(rng, 120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir, detail_dir = write_shards(root, rng, baseline.iloc[:90])
            baseline.to_csv(root / "base.csv", index=False)
            membership, subset, summary = subsets.select_random(baseline, [3, 4, 5, 6], 20, baseline_name="base.csv")
            subsets.write_batch(root / "batch", membership, subset, summary)
            out = root / "out"
            run.main(["--batch-dir", str(root / "batch"), "--baseline-csv", str(root / "base.csv"), "--global-dir", str(global_dir), "--detail-dir", str(detail_dir),
                      "--reselect-available", "--standardize", "--steps", "pooled,within,halves", "--splits", "2", "--permutations", "2", "--reduce-dim", "16",
                      "--trend-permutations", "10", "--halves-splits", "2", "--output-dir", str(out)])
            chosen = json.loads((out / "selection/summary.json").read_text())
            self.assertEqual(chosen["design"], "random")
            self.assertTrue(set(pd.read_csv(out / "selection/membership.csv")["story_key"]) <= set(baseline["story_key"].iloc[:90]))
            table = pd.read_csv(out / "halves_isolated/overlap_halves.csv")
            self.assertTrue(((table["overlap"] >= 0) & (table["overlap"] <= 1 + 1e-9)).all())
            self.assertTrue((out / "halves_isolated/fig_halves_overlap.png").exists())
            self.assertTrue((out / "pooled/fig_age_by_variance.png").exists())
            self.assertTrue((out / "fig_four_panels.png").exists())
            run.main(["--batch-dir", str(root / "batch"), "--baseline-csv", str(root / "base.csv"), "--global-dir", str(global_dir), "--detail-dir", str(detail_dir),
                      "--reselect-available", "--steps", "chunks", "--chunk-draws", "3", "--curve-permutations", "50", "--output-dir", str(out)])
            chunks = json.loads((out / "chunks_isolated/chunk_age_tests.json").read_text())
            self.assertIn("age_overlap_d1", chunks)
            self.assertTrue(np.isfinite(chunks["age_centroid_d1"]["T"]))
            self.assertTrue((out / "chunks_isolated/fig_chunks.png").exists())
            self.assertEqual(set(pd.read_csv(out / "chunks_isolated/chunks_by_age_distance.csv")["distance"]), {1, 2, 3, 4})
            for folder in ("pooled", "within_isolated"):
                self.assertTrue(np.isfinite(json.loads((out / folder / "age_top5_test.json").read_text())["T"]))
            self.assertTrue((out / "halves_isolated/age_top5_test.json").exists())
            (out / "fig_four_panels.png").unlink()
            run.main(["--batch-dir", str(root / "batch"), "--baseline-csv", str(root / "base.csv"), "--steps", "overview", "--output-dir", str(out)])
            self.assertTrue((out / "fig_four_panels.png").exists())

    def test_halves_alone_runs_and_halves_of_a_story_share_axes(self) -> None:
        rng = np.random.default_rng(14)
        baseline = make_baseline(rng, 100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir, detail_dir = write_shards(root, rng, baseline)
            baseline.to_csv(root / "base.csv", index=False)
            membership, subset, summary = subsets.select_random(baseline, [3, 4, 5], 40, baseline_name="base.csv")
            subsets.write_batch(root / "batch", membership, subset, summary)
            out = root / "out"
            run.main(["--batch-dir", str(root / "batch"), "--baseline-csv", str(root / "base.csv"), "--global-dir", str(global_dir), "--detail-dir", str(detail_dir),
                      "--steps", "halves", "--halves-splits", "2", "--output-dir", str(out)])
            self.assertFalse((out / "fig_spectrum_by_age.png").exists())
            table = pd.read_csv(out / "halves_isolated/overlap_halves.csv")
            self.assertTrue(((table["overlap"] >= 0) & (table["overlap"] <= 1 + 1e-9)).all())
            self.assertTrue((table["overlap"].dropna() >= table["null_mean"].dropna().min() - 1e-9).all())


class ChunkTests(unittest.TestCase):
    def test_chunk_count_is_fewer_for_short_stories(self) -> None:
        self.assertEqual([run.chunk_count(n, 5, 7) for n in (20, 21, 28, 35, 90)], [2, 3, 4, 5, 5])

    def test_story_with_position_drift_beats_random_chunks_and_a_shuffled_story_does_not(self) -> None:
        rng = np.random.default_rng(3)
        d, n = 40, 60
        t = np.linspace(0, 1, n)[:, None]
        drift = rng.normal(size=(2, d)) * 3
        ordered_story = np.sin(np.pi * t) * drift[0] + t * drift[1] + rng.normal(size=(n, d))  # a path through the embedding space
        shuffled_story = ordered_story[rng.permutation(n)]
        stats = run.story_chunk_stats(ordered_story, 5, 3, 20, np.random.default_rng(0))
        self.assertGreater(stats["ordered_centroid"][0], stats["random_centroid"][0] + 0.2)
        self.assertGreater(stats["ordered_centroid"][0], stats["ordered_centroid"][3])  # nearer chunks are more alike
        flat = run.story_chunk_stats(shuffled_story, 5, 3, 20, np.random.default_rng(0))
        self.assertLess(abs(flat["ordered_centroid"][0] - flat["random_centroid"][0]), 0.2)


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
                         "within_isolated/fig_within_and_pooled_within.png", "within_isolated/k_by_age_variance_within_pooled.csv",
                         "within_isolated/age_curve_test.json", "within_isolated/fig_age_curve_test.png", "pooled/age_curve_test.json"):
                self.assertTrue((out / name).exists(), name)
            k = pd.read_csv(out / "within_isolated/k_by_age_variance_within.csv")
            at_90 = k[np.isclose(k["variance"], 0.9)].set_index("age")["mean_k"]
            self.assertGreater(at_90.loc[6], at_90.loc[3])
            self.assertEqual(json.loads((out / "selection/summary.json").read_text())["cell_size"], 25)


if __name__ == "__main__":
    unittest.main()
