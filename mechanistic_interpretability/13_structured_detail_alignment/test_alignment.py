from __future__ import annotations

import importlib
import unittest
from argparse import Namespace

import numpy as np


alignment = importlib.import_module("mechanistic_interpretability.13_structured_detail_alignment.align")


class TokenEmbedder:
    vocabulary = ["cat", "left", "home", "dog", "searched", "park", "friends", "returned"]

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            words = text.lower().split()
            rows.append([words.count(token) for token in self.vocabulary])
        return alignment.normalize_rows(np.asarray(rows, dtype=np.float32))


class StructuredAlignmentTests(unittest.TestCase):
    def test_canonical_condition_field_is_recognized(self) -> None:
        self.assertEqual(alignment.condition_label({"condition": "older"}), "older")

    def test_summary_complete_monotonic_alignment(self) -> None:
        original = {
            "story_key": "story",
            "video_id": "video",
            "source_age": 8.0,
            "segments": [
                {"text": "cat left home"},
                {"text": "dog searched park"},
                {"text": "cat dog friends"},
                {"text": "cat returned home"},
            ],
        }
        summary = {
            "story_key": "story",
            "target_age": 8.0,
            "age_info_condition": "actual",
            "summary_id": 1,
            "segments": [
                {"text": "cat left home dog searched park"},
                {"text": "cat dog friends returned home"},
            ],
        }
        args = Namespace(
            max_span=3,
            top_candidates=8,
            length_penalty=0.01,
            overlap_penalty=0.1,
            reversal_penalty=1.0,
            null_similarity=0.2,
            source_capacity=2.0,
        )
        original_rows, summary_rows, story_row, window_rows = alignment.process_pair(
            original, summary, TokenEmbedder(), args
        )
        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(story_row["summary_coverage_rate"], 1.0)
        self.assertLessEqual(summary_rows[0]["source_start"], summary_rows[1]["source_start"])
        self.assertTrue(any(row["source_span_length"] > 1 for row in summary_rows))
        self.assertEqual({row["window_length"] for row in window_rows}, {1, 2, 3})
        self.assertEqual(len(original_rows), 4)
        self.assertIn("summary_centrality_rank", summary_rows[0])
        self.assertIn("centrality_rank_shift", summary_rows[0])
        self.assertIn("centrality_weighted_semantic_preservation", story_row)
        self.assertIn("retained_minus_omitted_semantic_match", story_row)
        self.assertIn("mean_centrality_rank_shift", story_row)

    def test_transport_can_choose_null(self) -> None:
        similarities = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float)
        source, null = alignment.capacitated_transport(
            similarities, null_similarity=0.5, source_capacity=1.0
        )
        self.assertAlmostEqual(float(source[:, 0].sum() + null[0]), 1.0)
        self.assertAlmostEqual(float(source[:, 1].sum() + null[1]), 1.0)
        self.assertEqual(int(np.argmax(source[:, 0])), 0)
        self.assertEqual(int(np.argmax(source[:, 1])), 1)

    def test_exact_duplicates_are_removed_but_variants_remain(self) -> None:
        base = {
            "story_key": "story",
            "age_info_condition": "actual",
            "summary_id": 1,
            "summary_text": "The cat returned.",
            "segments": [{"text": "The cat returned."}],
        }
        changed = dict(base, summary_text="The cat came home.", segments=[{"text": "The cat came home."}])
        rows = alignment.unique_summary_records([base, dict(base), changed])
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["_summary_variant"] for row in rows], [1, 2])
        self.assertNotEqual(rows[0]["_summary_text_hash"], rows[1]["_summary_text_hash"])


if __name__ == "__main__":
    unittest.main()
