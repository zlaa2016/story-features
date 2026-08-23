from __future__ import annotations

import importlib
import unittest

import numpy as np


comparison = importlib.import_module(
    "mechanistic_interpretability.17_covariance_detail_geometry.compare_models"
)


class CrossModelComparisonTests(unittest.TestCase):
    def test_linear_cka_is_invariant_to_orthogonal_rotation(self) -> None:
        rng = np.random.default_rng(23)
        first = rng.normal(size=(400, 12))
        rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        second = first @ rotation
        self.assertGreater(comparison.linear_cka(first, second), 0.999)

    def test_ridge_predictivity_recovers_shared_representation(self) -> None:
        rng = np.random.default_rng(29)
        first = rng.normal(size=(300, 10))
        transform = rng.normal(size=(10, 8))
        second = first @ transform + rng.normal(scale=0.01, size=(300, 8))
        stories = np.repeat([f"story_{index}" for index in range(60)], 5)
        forward, reverse = comparison.symmetric_ridge_predictivity(
            first, second, stories, folds=5, alpha=0.01
        )
        self.assertGreater(forward, 0.99)
        self.assertGreater(reverse, 0.70)

    def test_within_story_shuffle_never_crosses_story_boundary(self) -> None:
        values = np.arange(24).reshape(8, 3)
        stories = np.asarray(["a"] * 4 + ["b"] * 4)
        shuffled = comparison.shuffle_rows_within_story(values, stories, np.random.default_rng(31))
        self.assertEqual(set(shuffled[:4, 0]), set(values[:4, 0]))
        self.assertEqual(set(shuffled[4:, 0]), set(values[4:, 0]))


if __name__ == "__main__":
    unittest.main()
