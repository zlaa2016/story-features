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
detail_analysis = importlib.import_module(
    "mechanistic_interpretability.02_representation_extraction.analyze_detail_compression"
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

    def test_records_follow_global_shard_order(self) -> None:
        import pandas as pd

        records = [
            {"record_id": "first", "text": "one"},
            {"record_id": "second", "text": "two"},
            {"record_id": "third", "text": "three"},
        ]
        metadata = pd.DataFrame({"record_id": ["third", "first"]})
        selected = detail_controls.records_for_global_shard(records, metadata)
        self.assertEqual([record["record_id"] for record in selected], ["third", "first"])

    def test_missing_global_shard_record_fails(self) -> None:
        import pandas as pd

        with self.assertRaises(KeyError):
            detail_controls.records_for_global_shard(
                [{"record_id": "present"}], pd.DataFrame({"record_id": ["missing"]})
            )

    def test_isolated_detail_global_weight_change(self) -> None:
        import pandas as pd

        original_metadata = pd.DataFrame({"detail_index": [0, 1], "detail_text": ["first", "second"]})
        summary_metadata = pd.DataFrame({
            "record_id": ["summary", "summary"],
            "story_key": ["story", "story"],
            "condition": ["older", "older"],
            "summary_id": [1, 1],
            "source_age": [6, 6],
            "target_age": [9, 9],
            "detail_index": [0, 1],
            "detail_text": ["first summary", "second summary"],
        })
        original_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        summary_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        original_isolated = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        summary_isolated = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        summary_rows, original_rows, story_row = detail_analysis.process_summary(
            original_metadata,
            original_vectors,
            np.asarray([1.0, 0.0]),
            summary_metadata,
            summary_vectors,
            np.asarray([0.0, 1.0]),
            layer=3,
            representation="mean",
            original_isolated=original_isolated,
            summary_isolated=summary_isolated,
        )
        self.assertAlmostEqual(
            original_rows[0]["isolated_original_detail_global_weight_change"], -1.0
        )
        self.assertAlmostEqual(
            original_rows[1]["isolated_original_detail_global_weight_change"], 1.0
        )
        self.assertAlmostEqual(story_row["mean_isolated_original_detail_global_weight_change"], 0.0)
        self.assertIn("isolated_summary_detail_to_summary_global_cosine", summary_rows[0])


if __name__ == "__main__":
    unittest.main()
