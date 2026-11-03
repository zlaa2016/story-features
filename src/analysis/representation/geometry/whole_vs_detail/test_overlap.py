from __future__ import annotations

import unittest

import numpy as np

from analysis.representation.geometry.whole_vs_detail import overlap


def make(rng: np.random.Generator, n: int = 12, d: int = 64, signal: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A story: n detail vectors around a story direction, and a whole-story vector built from that direction plus noise."""
    direction = rng.normal(size=d)
    details = signal * direction + rng.normal(size=(n, d))
    whole = signal * direction * 0.5 + rng.normal(size=d) * 0.5
    return whole, details


class OverlapTests(unittest.TestCase):
    def test_related_story_scores_above_unrelated(self) -> None:
        rng = np.random.default_rng(0)
        related, unrelated = [], []
        for _ in range(30):
            whole, details = make(rng, signal=1.5)
            related.append(overlap.story_measures(whole, details, rng)[0]["cos_mean"])
            unrelated.append(overlap.story_measures(rng.normal(size=64), details, rng)[0]["cos_mean"])
        self.assertGreater(np.mean(related), 0.5)
        self.assertLess(abs(np.mean(unrelated)), 0.1)

    def test_span_excess_is_near_zero_for_a_random_direction_and_high_for_a_member(self) -> None:
        rng = np.random.default_rng(1)
        excess = []
        for _ in range(60):
            _, details = make(rng, n=10, d=200)
            measures, _ = overlap.story_measures(rng.normal(size=200), details, rng)
            excess.append(measures["r2_span_excess"])
        self.assertLess(abs(np.mean(excess)), 0.02)
        inside = details.T @ rng.normal(size=10)
        self.assertGreater(overlap.story_measures(inside, details, rng)[0]["r2_span_excess"], 0.9)

    def test_span_r2_handles_a_rank_deficient_set(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(size=(3, 40))
        details = np.vstack([base, base[0] + base[1]])           # 4 rows, rank 3
        r2, rank = overlap.span_r2(base[2], details)
        self.assertEqual(rank, 3)
        self.assertAlmostEqual(r2, 1.0, places=8)

    def test_prefix_curve_matches_a_hand_computation(self) -> None:
        details = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        whole = np.array([1.0, 0.0])
        curve = overlap.prefix_cosines(whole, details)
        self.assertAlmostEqual(curve[0], 1.0)
        self.assertAlmostEqual(curve[1], 1 / np.sqrt(2))
        self.assertAlmostEqual(curve[2], 0.0)

    def test_order_gain_is_positive_when_the_aligned_details_come_first(self) -> None:
        rng = np.random.default_rng(3)
        whole = np.zeros(30)
        whole[0] = 1.0
        aligned = np.tile(whole, (6, 1)) + rng.normal(scale=0.05, size=(6, 30))
        other = rng.normal(size=(6, 30)) * 0.8
        other[:, 0] = 0
        forward = overlap.story_measures(whole, np.vstack([aligned, other]), rng, draws=50)[0]["order_gain"]
        backward = overlap.story_measures(whole, np.vstack([other, aligned]), rng, draws=50)[0]["order_gain"]
        self.assertGreater(forward, 0.05)
        self.assertLess(backward, -0.05)

    def test_order_gain_is_about_zero_when_details_are_exchangeable(self) -> None:
        rng = np.random.default_rng(4)
        gains = [overlap.story_measures(*make(rng, n=20), rng, draws=30)[0]["order_gain"] for _ in range(40)]
        self.assertLess(abs(np.mean(gains)), 0.02)

    def test_identification_is_high_for_matched_pairs_and_chance_for_shuffled(self) -> None:
        rng = np.random.default_rng(5)
        stories = [make(rng, signal=1.5) for _ in range(80)]
        wholes = np.array([w for w, _ in stories])
        means = np.array([d.mean(axis=0) for _, d in stories])
        auc, _ = overlap.identification(wholes, means)
        self.assertGreater(auc.mean(), 0.9)
        shuffled, _ = overlap.identification(wholes, means[rng.permutation(80)])
        self.assertAlmostEqual(shuffled.mean(), 0.5, delta=0.08)

    def test_resample_keeps_endpoints_and_shape(self) -> None:
        curve = np.linspace(0, 1, 7)
        out = overlap.resample(curve, 10)
        self.assertEqual(out.shape, (10,))
        self.assertAlmostEqual(out[-1], 1.0)
        self.assertTrue(np.all(np.diff(out) >= -1e-12))


class PipelineTests(unittest.TestCase):
    def test_analyze_and_summarize_recover_a_planted_age_effect(self) -> None:
        import tempfile
        from pathlib import Path

        import pandas as pd

        from analysis.representation.geometry.whole_vs_detail import run

        rng = np.random.default_rng(6)
        ids, wholes, rows, vectors, membership, subset = [], [], [], [], [], []
        for s in range(90):
            age = 2 + s % 11
            key = f"v{s}__story01"
            n = int(rng.integers(6, 20))
            whole, details = make(rng, n=n, d=48, signal=0.3 + 0.12 * age)     # older stories: details say more about the whole
            ids.append(key)
            wholes.append(whole)
            for j in range(n):
                rows.append({"story_key": key, "detail_index": j})
                vectors.append(details[j])
            membership.append({"age": age, "story_key": key, "rank": 1})
            subset.append({"story_key": key, "range_lower": age - 1, "range_upper": age + 1})
        table, curves = run.analyze(np.array(ids), np.array(wholes), pd.DataFrame(rows), np.array(vectors), pd.DataFrame(membership), pd.DataFrame(subset))
        self.assertEqual(len(table), 90)
        self.assertEqual(curves.shape[0], 90)
        results = run.summarize(table, curves, pd.DataFrame(membership))
        self.assertTrue(all(float(c) > 0 for c in results["prefix"].columns[1:]))      # only position columns, not membership extras such as rank
        tests = results["tests"]
        midpoint = tests[tests["age_definition"].str.startswith("midpoint")].set_index("measure")
        self.assertGreater(midpoint.loc["cos_mean", "rho"], 0.7)            # planted: older stories' details say more about the whole
        self.assertGreater(midpoint.loc["cos_mean", "partial_rho"], 0.7)    # and not through the number of details
        self.assertLess(abs(midpoint.loc["order_gain", "rho"]), 0.35)       # no order structure was planted
        self.assertTrue((table["auc_identification"].between(0, 1)).all())
        with tempfile.TemporaryDirectory() as folder:
            run.plot_by_age(results["summary"], results["prefix"], Path(folder))
            self.assertGreater((Path(folder) / "fig_overlap_by_age.png").stat().st_size, 5000)
            self.assertGreater((Path(folder) / "fig_prefix_by_age.png").stat().st_size, 5000)

    def test_stories_with_too_few_details_are_dropped(self) -> None:
        import pandas as pd

        from analysis.representation.geometry.whole_vs_detail import run

        rng = np.random.default_rng(7)
        rows, vectors = [], []
        for key, n in (("a__story01", 3), ("b__story01", 8), ("c__story01", 9)):
            for j in range(n):
                rows.append({"story_key": key, "detail_index": j})
                vectors.append(rng.normal(size=16))
        membership = pd.DataFrame({"age": [3, 3, 4], "story_key": ["a__story01", "b__story01", "c__story01"]})
        subset = pd.DataFrame({"story_key": ["a__story01", "b__story01", "c__story01"], "range_lower": [2, 3, 3], "range_upper": [4, 5, 5]})
        table, _ = run.analyze(np.array(["a__story01", "b__story01", "c__story01"]), rng.normal(size=(3, 16)), pd.DataFrame(rows), np.array(vectors), membership, subset, min_details=4)
        self.assertEqual(sorted(table["story_key"]), ["b__story01", "c__story01"])


if __name__ == "__main__":
    unittest.main()
