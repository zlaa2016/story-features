from __future__ import annotations

import importlib
import unittest

import numpy as np


demixing = importlib.import_module("mechanistic_interpretability.08_demixing_normalization.analyze")


class DemixingTests(unittest.TestCase):
    def test_transformer_is_deterministic_and_fold_safe(self) -> None:
        rng = np.random.default_rng(4)
        training = rng.normal(size=(40, 12))
        evaluation = rng.normal(size=(7, 12))
        first = demixing.PcaSignedNmf(pca_dim=6, components=4, seed=9).fit(training)
        second = demixing.PcaSignedNmf(pca_dim=6, components=4, seed=9).fit(training)
        transformed = first.transform(evaluation)
        self.assertEqual(transformed.shape, (7, 4))
        self.assertTrue(np.all(transformed >= 0))
        np.testing.assert_allclose(transformed, second.transform(evaluation), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
