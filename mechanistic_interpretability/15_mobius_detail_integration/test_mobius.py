from __future__ import annotations

import importlib
import unittest

mobius_coefficients = importlib.import_module(
    "mechanistic_interpretability.15_mobius_detail_integration.compute"
).mobius_coefficients


class MobiusTests(unittest.TestCase):
    def test_additive_set_function_has_no_higher_order_terms(self) -> None:
        values = {(0,): 2.0, (1,): 3.0, (0, 1): 5.0}
        coefficients = mobius_coefficients(values)
        self.assertAlmostEqual(coefficients[(0,)], 2.0)
        self.assertAlmostEqual(coefficients[(1,)], 3.0)
        self.assertAlmostEqual(coefficients[(0, 1)], 0.0)

    def test_synergy_is_recovered(self) -> None:
        values = {(0,): 1.0, (1,): 1.0, (0, 1): 3.5}
        self.assertAlmostEqual(mobius_coefficients(values)[(0, 1)], 1.5)


if __name__ == "__main__":
    unittest.main()
