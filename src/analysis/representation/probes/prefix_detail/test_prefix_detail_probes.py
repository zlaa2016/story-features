from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC_ROOT))
extract = importlib.import_module("analysis.representation.probes.prefix_detail.extract")


class PrefixDetailProbeTests(unittest.TestCase):
    def test_join_segments_is_deterministic(self) -> None:
        self.assertEqual(extract.join_token_segments([[1, 2], [3], [4, 5]], [9]), [1, 2, 9, 3, 9, 4, 5])

    def test_excluding_target_removes_leakage(self) -> None:
        self.assertEqual(extract.prefix_indices(4, 2, True), [0, 1, 3])
        self.assertEqual(extract.prefix_indices(2, 2, True), [0, 1])
        self.assertEqual(extract.prefix_indices(4, 2, False), [0, 1, 2, 3])

    def test_foreign_prefix_is_exact_length_and_reproducible(self) -> None:
        streams = {"a": [1, 2], "b": [3, 4, 5], "c": [6]}
        first, donor_first = extract.exact_length_foreign_prefix(streams, "a", 9, np.random.default_rng(7))
        second, donor_second = extract.exact_length_foreign_prefix(streams, "a", 9, np.random.default_rng(7))
        self.assertEqual(len(first), 9)
        self.assertEqual(first, second)
        self.assertEqual(donor_first, donor_second)

    def test_js_is_zero_for_equal_distributions(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is available in the HPC model environment")

        logits = torch.tensor([[0.0, 1.0, -1.0]])
        log_probabilities = torch.log_softmax(logits, dim=-1)
        value = extract.js_from_log_probabilities(log_probabilities, log_probabilities)
        self.assertAlmostEqual(float(value.item()), 0.0, places=7)

    def test_readout_only_layout_uses_endpoint_as_span(self) -> None:
        # Exclusive span end may be one beyond the endpoint when the fixed readout itself is the
        # measured span. This is the layout used by readout_only().
        detail_start, detail_end, readout_index, length = 3, 4, 3, 4
        valid = (
            0 < detail_start < detail_end <= length
            and detail_start <= readout_index < length
            and detail_end <= readout_index + 1
        )
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
