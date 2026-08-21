from __future__ import annotations

import importlib
import unittest

import pandas as pd

central_derivative = importlib.import_module(
    "mechanistic_interpretability.16_causal_compression_geometry.analyze"
).central_derivative


class GeometryTests(unittest.TestCase):
    def test_central_derivative(self) -> None:
        frame = pd.DataFrame({"alpha": [-1.0, 0.0, 1.0], "score": [0.2, 0.4, 0.8]})
        result = central_derivative(frame, "score")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["derivative"], 0.3)
        self.assertAlmostEqual(result["baseline"], 0.4)


if __name__ == "__main__":
    unittest.main()
