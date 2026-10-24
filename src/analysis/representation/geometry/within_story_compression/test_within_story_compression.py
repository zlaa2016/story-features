from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


module = importlib.import_module("analysis.representation.geometry.within_story_compression.analyze")


def random_story(rng: np.random.Generator, n: int, d: int, rank: int, trend: float = 0.0) -> np.ndarray:
    loadings = rng.normal(size=(rank, d))
    scores = rng.normal(size=(n, rank))
    values = scores @ loadings
    if trend:
        direction = rng.normal(size=d)
        values = values + trend * np.linspace(-1, 1, n)[:, None] * direction[None, :] * np.sqrt(d)
    return values


class SpectrumTests(unittest.TestCase):
    def test_known_spectrum(self) -> None:
        values = np.array([8.0, 4.0, 2.0, 1.0, 1.0])
        stats = module.spectrum_stats(values, 6)
        self.assertAlmostEqual(stats["head1"], 0.5)
        self.assertAlmostEqual(stats["head3"], 14 / 16)
        self.assertEqual(stats["k50"], 1.0)
        self.assertEqual(stats["k90"], 4.0)
        self.assertAlmostEqual(stats["kn90"], 4.0 / 5.0)

    def test_flat_spectrum_has_participation_ratio_equal_to_rank(self) -> None:
        stats = module.spectrum_stats(np.ones(7), 8)
        self.assertAlmostEqual(stats["pr"], 7.0)
        self.assertLess(abs(stats["alpha"]), 1e-9)

    def test_power_law_exponent_recovered(self) -> None:
        values = np.arange(1, 31, dtype=float) ** -1.5
        stats = module.spectrum_stats(values, 31)
        self.assertAlmostEqual(stats["alpha"], 1.5, places=6)


class PrefixTests(unittest.TestCase):
    def test_new_dimensions_raise_prefix_participation_and_repeats_do_not(self) -> None:
        rng = np.random.default_rng(0)
        fresh = rng.normal(size=(30, 80))
        repeated = np.tile(rng.normal(size=(3, 80)), (10, 1)) + 0.01 * rng.normal(size=(30, 80))
        sizes = [10, 20, 30]
        fresh_pr = module.prefix_participation(fresh @ fresh.T, sizes)
        repeated_pr = module.prefix_participation(repeated @ repeated.T, sizes)
        self.assertGreater(fresh_pr[2], fresh_pr[0] + 5)
        self.assertLess(repeated_pr[2], 4.0)

    def test_saturating_fit_recovers_time_constant(self) -> None:
        sizes = np.arange(5, 80, 5)
        participation = 12.0 * (1 - np.exp(-sizes / 15.0))
        amplitude, tau = module.saturating_fit(sizes, participation, 80)
        self.assertAlmostEqual(amplitude, 12.0, places=2)
        self.assertAlmostEqual(tau, 15.0, places=1)


class PositionTests(unittest.TestCase):
    def test_trend_is_detected_and_removed(self) -> None:
        rng = np.random.default_rng(1)
        n, d = 60, 40
        values = random_story(rng, n, d, rank=6, trend=1.5)
        zc = values - values.mean(axis=0)
        gram = module.center_gram(zc @ zc.T)
        basis = module.position_basis(n, 3)
        explained = module.position_fraction(gram, basis)
        shuffled = []
        for _ in range(20):
            order = rng.permutation(n)
            shuffled.append(module.position_fraction(gram[np.ix_(order, order)], basis))
        self.assertGreater(explained, 0.4)
        self.assertLess(np.mean(shuffled), 0.2)
        detrended = module.detrend_gram(gram, basis)
        self.assertLess(module.position_fraction(detrended, basis), 1e-8)
        correlation, rank = module.leading_position_correlation(gram)
        self.assertGreater(correlation, 0.8)
        self.assertEqual(rank, 1)

    def test_no_trend_has_small_position_share(self) -> None:
        rng = np.random.default_rng(2)
        values = random_story(rng, 60, 40, rank=6)
        zc = values - values.mean(axis=0)
        gram = module.center_gram(zc @ zc.T)
        self.assertLess(module.position_fraction(gram, module.position_basis(60, 3)), 0.25)


class AssignmentTests(unittest.TestCase):
    def test_assignment_respects_ranges_and_capacity(self) -> None:
        rng = np.random.default_rng(3)
        lo = np.array([2, 2, 3, 4, 4, 4, 5, 8, 9, 12])
        hi = np.array([2, 3, 12, 6, 4, 12, 5, 9, 12, 12])
        ages = np.arange(2, 13)
        assigned = module.assign_ages(lo, hi, ages, per_age=2, rng=rng)
        self.assertFalse(np.isnan(assigned).any())
        self.assertTrue(((assigned >= lo) & (assigned <= hi)).all())
        _, counts = np.unique(assigned, return_counts=True)
        self.assertLessEqual(counts.max(), 2)

    def test_story_with_no_free_age_is_left_unassigned(self) -> None:
        lo = np.array([5, 5, 5])
        hi = np.array([5, 5, 5])
        assigned = module.assign_ages(lo, hi, np.arange(2, 13), per_age=2, rng=np.random.default_rng(0))
        self.assertEqual(int(np.isfinite(assigned).sum()), 2)

    def test_equalize_trims_to_smallest(self) -> None:
        assigned = np.array([3, 3, 3, 4, 4, 5], dtype=float)
        trimmed = module.equalize(assigned, np.random.default_rng(0))
        _, counts = np.unique(trimmed[np.isfinite(trimmed)], return_counts=True)
        self.assertEqual(set(counts.tolist()), {1})


class AssociationTests(unittest.TestCase):
    def frame(self, direct_age_effect: float) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        count = 400
        age = rng.integers(2, 13, size=count).astype(float)
        size = np.exp(3.0 + 0.12 * age + rng.normal(scale=0.4, size=count))
        stat = -1.0 * np.log(size) + direct_age_effect * age + rng.normal(scale=0.5, size=count)
        return pd.DataFrame({"age": age, "n_units": size, "stat": stat})

    def test_size_driven_trend_is_not_reported_as_an_age_effect(self) -> None:
        result = module.age_association(self.frame(0.0), "stat", np.random.default_rng(0), 400, bootstrap=100)
        self.assertLess(result["rho"], -0.2)
        self.assertLess(abs(result["partial_rho"]), 0.15)
        self.assertGreater(result["p_partial"], 0.05)

    def test_direct_age_effect_survives_size_adjustment(self) -> None:
        result = module.age_association(self.frame(0.15), "stat", np.random.default_rng(0), 400, bootstrap=100)
        self.assertGreater(result["partial_rho"], 0.25)
        self.assertLess(result["p_partial"], 0.01)
        self.assertGreater(result["partial_ci_low"], 0.0)


class AgeProfileTests(unittest.TestCase):
    def test_profile_recovers_converging_curves(self) -> None:
        rng = np.random.default_rng(11)
        count = 300
        age = rng.integers(2, 13, size=count).astype(float)
        size = np.exp(rng.normal(4.0, 0.3, size=count))
        data = {"age": age, "n_units": size}
        for v in module.V_GRID:
            label = int(round(v * 100))
            effect = 0.10 * (v - 0.20) / 0.75
            log_k = np.log(size) * 0.5 + effect * age + rng.normal(scale=0.05, size=count)
            data[f"k{label}"] = np.exp(log_k)
            data[f"kr{label}"] = np.exp(log_k - 0.5 * np.log(size))
        profile = module.age_profile(pd.DataFrame(data), np.random.default_rng(0), 100)
        raw = profile[profile["family"] == "raw_k"].set_index("v")["beta_age_per_year"]
        self.assertAlmostEqual(raw.loc[0.95], 0.10, delta=0.02)
        self.assertAlmostEqual(raw.loc[0.2], 0.0, delta=0.02)
        self.assertGreater(raw.loc[0.95] - raw.loc[0.2], 0.06)

    def test_grid_reaches_95_percent(self) -> None:
        self.assertAlmostEqual(float(module.V_GRID[-1]), 0.95)
        stats = module.spectrum_stats(np.ones(20), 21)
        self.assertEqual(stats["k95"], 19.0)


class EndToEndTests(unittest.TestCase):
    def test_synthetic_shards_run_through_the_cli(self) -> None:
        rng = np.random.default_rng(4)
        d, layers = 24, np.array([0, 23], dtype=np.int16)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            shard_dir = root_path / "isolated"
            shard_dir.mkdir()
            rows, vectors = [], []
            story_number = 0
            for age in (3, 6, 9):
                for _ in range(14):
                    n = int(rng.integers(25, 40))
                    rank = 2 + age // 3
                    values = random_story(rng, n, d, rank=rank)
                    for detail_index in range(n):
                        rows.append({
                            "record_id": f"s{story_number}::original", "record_type": "original",
                            "story_key": f"s{story_number}", "video_id": f"v{story_number}",
                            "source_age": float(age), "detail_index": detail_index,
                        })
                    vectors.append(values)
                    story_number += 1
            stacked = np.concatenate(vectors).astype(np.float16)
            per_layer = np.stack([stacked, stacked * 0.5], axis=1)
            np.savez(shard_dir / "isolated_detail_features_000.npz", mean_vectors=per_layer, layers=layers)
            pd.DataFrame(rows).to_csv(shard_dir / "isolated_detail_metadata_000.csv", index=False)
            output = root_path / "out"
            module.main([
                "--isolated-dir", str(shard_dir), "--views", "isolated",
                "--age-min", "3", "--age-max", "9", "--per-age", "10", "--min-units", "20",
                "--null-draws", "3", "--permutations", "50", "--bootstrap", "50",
                "--n-assignments", "2", "--output-dir", str(output),
            ])
            view_dir = output / "isolated"
            for name in ("story_profiles.csv", "assignments.csv", "age_summary_r00.csv", "age_tests_r00.csv",
                         "age_tests_across_assignments.csv", "hypothesis_check_r00.json", "fig_k_curves.png",
                         "fig_incremental_pr.png", "fig_position_drift.png", "run_provenance.json"):
                self.assertTrue((view_dir / name).exists(), name)
            for name in ("k_age_profile_r00.csv", "fig_k_shape_by_age_group.png", "fig_k_age_profile.png"):
                self.assertTrue((view_dir / name).exists(), name)
            fixed_output = root_path / "fixed"
            module.main([
                "--isolated-dir", str(shard_dir), "--views", "isolated",
                "--age-min", "3", "--age-max", "9", "--per-age", "10", "--min-units", "20", "--fixed-units", "25",
                "--null-draws", "3", "--permutations", "30", "--bootstrap", "30",
                "--age-groups", "3-4,5-6,7-9", "--output-dir", str(fixed_output),
            ])
            fixed_profiles = pd.read_csv(fixed_output / "isolated" / "story_profiles.csv")
            self.assertEqual(set(fixed_profiles["n_units"].astype(int)), {25})
            summary = pd.read_csv(view_dir / "age_summary_r00.csv")
            self.assertEqual(sorted(summary["age"].unique().tolist()), [3.0, 6.0, 9.0])
            self.assertEqual(int(summary[summary["stat"] == "k90"]["n_stories"].max()), 10)
            tests = pd.read_csv(view_dir / "age_tests_r00.csv")
            effective_rank = tests[tests["stat"] == "effrank"].iloc[0]
            self.assertGreater(effective_rank["rho"], 0.3)


if __name__ == "__main__":
    unittest.main()
