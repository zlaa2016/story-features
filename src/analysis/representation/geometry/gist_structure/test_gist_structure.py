from __future__ import annotations

import argparse
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

gm = importlib.import_module("analysis.representation.geometry.gist_structure.gist_math")
select = importlib.import_module("analysis.representation.geometry.gist_structure.select_stories")
within = importlib.import_module("analysis.representation.geometry.gist_structure.within_story")
schema = importlib.import_module("analysis.representation.geometry.gist_structure.schema_age")
calibrate = importlib.import_module("analysis.representation.geometry.gist_structure.calibrate")


def gist_story(rng: np.random.Generator, n: int, gists: int, noise: float, d: int = 128, blocky: bool = True) -> np.ndarray:
    centers = rng.normal(size=(gists, d))
    if blocky and gists > 1:
        assignment = np.zeros(n, dtype=int)
        for cut in np.sort(rng.choice(np.arange(1, n), gists - 1, replace=False)):
            assignment[cut:] += 1
    else:
        assignment = rng.integers(0, gists, n)
    return centers[assignment] + noise * rng.normal(size=(n, d))


class GistMathTests(unittest.TestCase):
    def test_held_out_curve_saturates_at_the_number_of_gists_and_ignores_noise_only_data(self) -> None:
        rng = np.random.default_rng(0)
        for gists in (3, 6):
            z = gist_story(rng, 80, gists, 0.6)
            _, _, excess = gm.excess_curve(gm.oos_pca_curve, z, rng, 3)
            scalars = gm.saturation_scalars(excess, gm.M_GRID, 80, "gist")
            self.assertGreater(scalars["gist_peak"], 0.3)
            self.assertLessEqual(abs(scalars["gist_count"] - (gists - 1)), 2)
        noise_only = rng.normal(size=(80, 128))
        _, _, excess = gm.excess_curve(gm.oos_pca_curve, noise_only, rng, 3)
        scalars = gm.saturation_scalars(excess, gm.M_GRID, 80, "gist")
        self.assertLess(scalars["gist_peak"], gm.MIN_PEAK + 0.03)

    def test_noise_changes_strength_but_not_the_gist_count(self) -> None:
        rng = np.random.default_rng(1)
        counts, peaks = [], []
        for noise in (0.5, 1.5):
            z = gist_story(rng, 90, 5, noise)
            _, _, excess = gm.excess_curve(gm.oos_pca_curve, z, rng, 3)
            scalars = gm.saturation_scalars(excess, gm.M_GRID, 90, "gist")
            counts.append(scalars["gist_count"])
            peaks.append(scalars["gist_peak"])
        self.assertGreater(peaks[0], peaks[1])
        self.assertLessEqual(abs(counts[0] - counts[1]), 2)

    def test_event_structure_needs_contiguity(self) -> None:
        rng = np.random.default_rng(2)
        blocky = gist_story(rng, 60, 5, 0.6, blocky=True)
        scattered = gist_story(rng, 60, 5, 0.6, blocky=False)
        _, _, blocky_excess = gm.event_excess(blocky @ blocky.T, rng, 5)
        _, _, scattered_excess = gm.event_excess(scattered @ scattered.T, rng, 5)
        self.assertGreater(np.nanmax(blocky_excess), np.nanmax(scattered_excess) + 0.1)

    def test_vq_curve_gains_over_the_mean_when_there_are_clusters(self) -> None:
        rng = np.random.default_rng(3)
        z = gist_story(rng, 60, 4, 0.5, d=64)
        curve = gm.vq_curve(z, rng, folds=4, ms=(2, 4))
        self.assertGreater(curve[1], curve[0] - 0.05)
        self.assertGreater(curve[1], 0.2)

    def test_deduplication_removes_exact_repeats(self) -> None:
        rng = np.random.default_rng(4)
        rows = base_rows = rng.normal(size=(10, 32))
        rows = np.vstack([base_rows, base_rows[:4], base_rows[2:5]])
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        keep = gm.dedupe_keep(rows, 0.95)
        self.assertEqual(int(keep.sum()), 10)
        pool = rng.normal(size=(500, 32))
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        threshold = gm.duplicate_threshold(pool, np.arange(500) % 50, rng)
        self.assertTrue(0.8 <= threshold <= 0.98)

    def test_count_is_undefined_without_shared_structure(self) -> None:
        scalars = gm.saturation_scalars(np.full(len(gm.M_GRID), 0.005), gm.M_GRID, 50, "gist")
        self.assertTrue(np.isnan(scalars["gist_count"]))


class SelectionTests(unittest.TestCase):
    def test_overlap_and_primary_band(self) -> None:
        bands = select.parse_bands("2-3,4-5,6-7")
        lo = np.array([2, 4, 3, 2, 6])
        hi = np.array([3, 5, 4, 12, 7])
        fractions = select.overlap_fractions(lo, hi, bands)
        np.testing.assert_allclose(fractions[2], [0.5, 0.5, 0.0])
        chosen, best = select.assign_primary_band(lo, hi, bands, 0.5, np.random.default_rng(0))
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[1], 1)
        self.assertEqual(chosen[3], -1)
        self.assertEqual(chosen[4], 2)

    def test_quotas_sum_to_target_and_respect_availability(self) -> None:
        quota = select.allocate_quotas(np.array([30, 2, 30, 5]), 40, np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertEqual(int(quota.sum()), 40)
        self.assertLessEqual(int(quota[1]), 2)
        self.assertLessEqual(int(quota[3]), 5)

    def test_facility_location_covers_both_clusters_and_respects_author_cap(self) -> None:
        points = np.vstack([np.tile([1.0, 0.0], (5, 1)), np.tile([0.0, 1.0], (5, 1))]) + 0.01 * np.random.default_rng(0).normal(size=(10, 2))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        similarity = (points @ points.T + 1) / 2
        authors = np.array(["a", "a", "b", "c", "d", "e", "e", "f", "g", "h"])
        chosen = select.facility_location(similarity, np.ones(10), 3, np.zeros(10, dtype=bool), authors, 1,
                                          np.array(["x"] * 10), 10, {}, {})
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len({authors[i] for i in chosen}), 3)
        self.assertTrue(any(i < 5 for i in chosen) and any(i >= 5 for i in chosen))

    def test_selection_equalises_bands_and_matches_topic_mix(self) -> None:
        rng = np.random.default_rng(5)
        bands = select.parse_bands("2-3,4-5,6-7")
        rows = []
        for band_index in range(3):
            for number in range(60):
                topic = int(rng.choice(3, p=[0.2, 0.3, 0.5] if band_index == 0 else [0.5, 0.3, 0.2]))
                rows.append({"story_key": f"s{band_index}-{number}", "band_index": band_index, "band": bands[band_index][0],
                             "band_mid": float(band_index), "topic_cluster": topic, "popularity": float(rng.random()),
                             "author_group": f"author{band_index}-{number}", "channel": f"c{number % 6}"})
        candidates = pd.DataFrame(rows)
        means = rng.normal(size=(len(candidates), 8))
        means /= np.linalg.norm(means, axis=1, keepdims=True)
        result = select.select_stories(candidates, means, bands, 30, "popular_coverage", "match", 1, 0.4, True, rng)
        chosen = result[result["selected"]]
        self.assertEqual(set(chosen.groupby("band").size()), {30})
        mix = pd.crosstab(chosen["band"], chosen["topic_cluster"], normalize="index")
        self.assertLess(float((mix.max() - mix.min()).max()), 0.15)


class SchemaTests(unittest.TestCase):
    def nested(self, dims: list[int], rng: np.random.Generator, stories: int = 30, details: int = 24, reduced: int = 24) -> tuple[np.ndarray, np.ndarray]:
        basis = np.linalg.qr(rng.normal(size=(reduced, reduced)))[0]
        blocks, labels = [], []
        for band, dimension in enumerate(dims):
            for _ in range(stories):
                scores = rng.normal(size=(details, dimension))
                rows = scores @ basis[:dimension] + 0.15 * rng.normal(size=(details, reduced))
                blocks.append(rows - rows.mean(axis=0))
                labels.append(f"b{band}")
        return np.stack(blocks), np.array(labels)

    def config(self) -> argparse.Namespace:
        return argparse.Namespace(folds=4, null_draws=1, transfer_m=[2, 4, 8], bootstrap=5, permutations=5)

    def test_nested_subspaces_give_rising_dimension_and_negative_asymmetry(self) -> None:
        rng = np.random.default_rng(6)
        stories, labels = self.nested([3, 6, 10], rng)
        names = ["b0", "b1", "b2"]
        stats = schema.schema_stats(stories, labels, names, {"b0": 3.0, "b1": 6.0, "b2": 9.0}, rng, self.config(), lite=True)
        dims = [stats["bands"][name]["D80"] for name in names]
        self.assertLess(dims[0], dims[2])
        self.assertGreater(stats["slope_D80"], 0)
        self.assertLess(stats["asymmetry_mean"], 0)
        self.assertLess(stats["bands"]["b0"]["pa_dim"], stats["bands"]["b2"]["pa_dim"])

    def test_identical_bands_show_no_trend(self) -> None:
        rng = np.random.default_rng(7)
        stories, labels = self.nested([6, 6, 6], rng)
        names = ["b0", "b1", "b2"]
        stats = schema.schema_stats(stories, labels, names, {"b0": 3.0, "b1": 6.0, "b2": 9.0}, rng, self.config(), lite=True)
        self.assertLess(abs(stats["asymmetry_mean"]), 0.08)
        self.assertLess(abs(stats["slope_D80"]), 0.7)


class CalibrationTests(unittest.TestCase):
    def test_small_calibration_detects_planted_compressibility_and_not_the_null(self) -> None:
        effects = calibrate.run_calibration(["null", "compressibility"], 8, 96, 0, 11, log=lambda _: None)
        table = effects[(effects["model"] == "M1") & (effects["outcome"] == "gist_count_rel")].set_index("scenario")
        self.assertLess(table.loc["compressibility", "beta_std_per_year"], -0.1)
        self.assertLess(abs(table.loc["null", "beta_std_per_year"]), table.loc["compressibility", "beta_std_per_year"].__abs__())


def write_synthetic_shards(root: Path, rng: np.random.Generator, per_band: int = 14, d: int = 32) -> tuple[Path, Path, Path]:
    shard_dir = root / "isolated"
    shard_dir.mkdir()
    bands = [("2-3", 2, 3), ("4-5", 4, 5), ("6-7", 6, 7), ("8-12", 8, 12)]
    themes = rng.normal(size=(3, d)) * 2
    rows, vectors, master, flat = [], [], [], []
    number = 0
    for band_index, (_, low, high) in enumerate(bands):
        for _ in range(per_band):
            n = int(rng.integers(30, 45))
            topic = int(rng.integers(0, 3))
            gists = 3 + 2 * band_index
            values = themes[topic][None, :] + gist_story(rng, n, gists, 0.8, d=d)
            words = " ".join("".join(rng.choice(list("abcdefghij"), 4)) for _ in range(n * 6))
            for detail in range(n):
                rows.append({"record_id": f"s{number}::original", "record_type": "original", "story_key": f"s{number}",
                             "video_id": f"v{number}", "source_age": (low + high) / 2.0, "detail_index": detail,
                             "detail_text": " ".join(words.split()[detail * 6:(detail + 1) * 6]), "token_count": 8})
            vectors.append(values)
            master.append({"video_id": f"v{number}", "yt_view_count": float(rng.integers(100, 100000)), "Num Reviews": float(rng.integers(0, 500)),
                           "Author": f"author{number}", "Title": f"title{number}", "yt_channel": f"channel{number % 5}"})
            flat.append({"video_id": f"v{number}", "range_lower": low, "range_upper": high})
            number += 1
    stacked = np.concatenate(vectors).astype(np.float16)
    np.savez(shard_dir / "isolated_detail_features_000.npz", mean_vectors=np.stack([stacked, stacked * 0.5], axis=1), layers=np.array([0, 23], dtype=np.int16))
    pd.DataFrame(rows).to_csv(shard_dir / "isolated_detail_metadata_000.csv", index=False)
    pd.DataFrame(master).to_csv(root / "master.csv", index=False)
    pd.DataFrame(flat).to_csv(root / "flat.csv", index=False)
    return shard_dir, root / "master.csv", root / "flat.csv"


class RealControlTests(unittest.TestCase):
    def test_merge_adds_counts_and_halves_agree(self) -> None:
        rng = np.random.default_rng(12)
        cfg = argparse.Namespace(folds=5, null_draws=2, vq=False, vq_folds=3, dedupe=False, min_units=10)
        matrices, rows = {}, []
        for number in range(24):
            gists = 3 + (number % 4) * 2
            raw = gist_story(rng, 70, gists, 0.7, d=96)
            z = raw / (raw.std(axis=0) + 1e-6)
            matrices[f"s{number}"] = (z, z)
            measures = within.unit_measures(z, rng, cfg)
            rows.append({"story_key": f"s{number}", "band": "a" if number < 12 else "b", **measures})
        frame = pd.DataFrame(rows)
        merges, reliability, summary = within.real_data_controls(matrices, frame, cfg, rng, per_band=8)
        self.assertGreater(summary["merge_count_over_sum_median"], 0.6)
        self.assertLess(summary["merge_count_over_sum_median"], 1.5)
        self.assertGreater(summary["event_peak_ordered_minus_shuffled_median"], 0.0)
        gist = reliability.set_index("measure").loc["gist_count"]
        self.assertGreater(gist["split_half_rho"], 0.3)


class PipelineTests(unittest.TestCase):
    def test_select_then_within_story_then_schema_on_synthetic_shards(self) -> None:
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir, master, flat = write_synthetic_shards(root, rng)
            selection_dir = root / "selection"
            select.main([
                "--isolated-dir", str(shard_dir), "--master-csv", str(master), "--subset-csv", str(flat),
                "--bands", "2-3,4-5,6-7,8-12", "--per-band", "10", "--topic-clusters", "3", "--min-units", "20",
                "--output-dir", str(selection_dir),
            ])
            selected = pd.read_csv(selection_dir / "selected_stories.csv")
            self.assertEqual(set(selected.groupby("band").size()), {10})
            self.assertTrue((selection_dir / "balance_report.csv").exists())

            all_dir = root / "selection_all"
            select.main([
                "--isolated-dir", str(shard_dir), "--master-csv", str(master), "--subset-csv", str(flat),
                "--bands", "2-3,4-5,6-7,8-12", "--topic-clusters", "3", "--min-units", "20", "--mode", "all",
                "--output-dir", str(all_dir),
            ])
            everything = pd.read_csv(all_dir / "selected_stories.csv")
            self.assertEqual(len(everything), 56)
            self.assertTrue((everything["band_mid"] != everything["band_mid"].round()).any() or everything["band_mid"].nunique() > 1)

            within_dir = root / "within"
            within.main([
                "--isolated-dir", str(shard_dir), "--selected-csv", str(selection_dir / "selected_stories.csv"),
                "--min-units", "20", "--no-vq", "--bootstrap", "30", "--permutations", "30", "--folds", "5",
                "--fixed-units", "0,25", "--control-pairs", "6", "--output-dir", str(within_dir),
            ])
            for label in ("all_units", "fixed_25"):
                for name in ("story_gist_profiles.csv", "age_effects.csv", "band_summary.csv", "hypothesis_check.json",
                             "fig_gist_curves_by_band.png"):
                    self.assertTrue((within_dir / label / name).exists(), f"{label}/{name}")
            for name in ("real_data_merge_control.csv", "split_half_reliability.csv", "real_data_validation.json", "inclusion_by_band.csv"):
                self.assertTrue((within_dir / "all_units" / name).exists(), name)
            profiles = pd.read_csv(within_dir / "fixed_25" / "story_gist_profiles.csv")
            self.assertEqual(set(profiles["n_units"].astype(int)), {25})
            effects = pd.read_csv(within_dir / "all_units" / "age_effects.csv")
            self.assertIn("gist_count_rel", set(effects["outcome"]))
            self.assertEqual({"M0", "M1", "M2", "M3"} & set(effects["model"]), {"M0", "M1", "M2", "M3"})

            schema_dir = root / "schema"
            schema.main([
                "--isolated-dir", str(shard_dir), "--selected-csv", str(selection_dir / "selected_stories.csv"),
                "--details-per-story", "20", "--reduce-dim", "16", "--folds", "3", "--bootstrap", "4", "--permutations", "4",
                "--transfer-m", "2,4,8", "--effect-draws", "20", "--effect-permutations", "20", "--output-dir", str(schema_dir),
            ])
            for name in ("schema_band_curves.csv", "schema_dimensions.csv", "schema_dimension_bootstrap.csv", "schema_transfer.csv",
                         "schema_trend_tests.json", "story_schema_share.csv", "schema_story_age_effects.csv", "fig_schema_dimensionality.png"):
                self.assertTrue((schema_dir / name).exists(), name)
            dimensions = pd.read_csv(schema_dir / "schema_dimensions.csv")
            self.assertEqual(len(dimensions), 4)


if __name__ == "__main__":
    unittest.main()
