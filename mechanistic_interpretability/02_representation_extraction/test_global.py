from __future__ import annotations

import importlib
import unittest

import numpy as np


global_extraction = importlib.import_module(
    "mechanistic_interpretability.02_representation_extraction.extract_global"
)
global_analysis = importlib.import_module(
    "mechanistic_interpretability.02_representation_extraction.analyze_global"
)
detail_controls = importlib.import_module(
    "mechanistic_interpretability.02_representation_extraction.extract_detail_controls"
)


class GlobalRepresentationTests(unittest.TestCase):
    def test_negative_one_resolves_to_final_layer(self) -> None:
        self.assertEqual(global_extraction.resolve_requested_layers(24, "0,4,-1"), [0, 4, 23])

    def test_detail_deletion_preserves_remaining_order(self) -> None:
        text = "First detail. Second detail. Third detail."
        start = text.index("Second detail.")
        result = global_extraction.delete_span(text, (start, start + len("Second detail.")))
        self.assertEqual(result, "First detail. Third detail.")

    def test_repeated_segments_are_located_monotonically(self) -> None:
        text = "The cat ran. The dog waited. The cat ran."
        segments = [{"text": "The cat ran."}, {"text": "The dog waited."}, {"text": "The cat ran."}]
        spans = global_extraction.locate_segment_spans(text, segments)
        self.assertLess(spans[0][0], spans[1][0])
        self.assertLess(spans[1][0], spans[2][0])

    def test_token_indices_overlap_detail_span(self) -> None:
        offsets = [(0, 0), (0, 5), (6, 12), (12, 13), (0, 0)]
        self.assertEqual(global_extraction.token_indices_for_span(offsets, (6, 13)), [2, 3])

    def test_cosine_rows(self) -> None:
        left = np.asarray([[1.0, 0.0], [1.0, 1.0]])
        right = np.asarray([[1.0, 0.0], [-1.0, -1.0]])
        result = global_analysis.cosine_rows(left, right)
        np.testing.assert_allclose(result, [1.0, -1.0])

    def test_detail_subsample_includes_endpoints(self) -> None:
        self.assertEqual(detail_controls.selected_detail_indices(10, 4), [0, 3, 6, 9])

    def test_detail_subsample_keeps_all_when_below_limit(self) -> None:
        self.assertEqual(detail_controls.selected_detail_indices(3, 4), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
