from __future__ import annotations

import unittest

from shared.detail_metrics import lexical_preservation, surface_characteristics


class DetailMetricsTests(unittest.TestCase):
    def test_lexical_identity_is_one(self) -> None:
        result = lexical_preservation("The cat ran home.", "The cat ran home.")
        self.assertEqual(result["lexical_token_f1"], 1.0)
        self.assertEqual(result["lexical_lcs_f1"], 1.0)

    def test_lcs_penalizes_reordering(self) -> None:
        result = lexical_preservation("one two three", "three two one")
        self.assertEqual(result["lexical_token_f1"], 1.0)
        self.assertLess(result["lexical_lcs_f1"], 1.0)

    def test_surface_metrics_are_finite_for_text(self) -> None:
        result = surface_characteristics("First, she felt sad because it rained. Then she went home!")
        self.assertEqual(result["sentence_count"], 2)
        self.assertGreater(result["causal_connective_rate"], 0)
        self.assertGreater(result["temporal_connective_rate"], 0)


if __name__ == "__main__":
    unittest.main()
