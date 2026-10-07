from __future__ import annotations

import importlib
import unittest


validation = importlib.import_module("features.summaries.dataset_index.validate")


def record(record_type: str, condition: str, record_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "story_key": "story-1",
        "condition": condition,
        "target_age": 8,
        "compression_ratio": 1.0 if record_type == "original" else 0.3,
        "text": "The cat returned home.",
        "segments": [{"segment_id": 1, "text": "The cat returned home."}],
    }


class CohortValidationTests(unittest.TestCase):
    def test_valid_original_and_summary(self) -> None:
        report = validation.validate_records(
            [record("original", "original", "story-1::original"), record("summary", "actual", "story-1::actual::1")],
            3,
            18,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["segment_mismatch_count"], 0)

    def test_duplicate_record_and_bad_segments_fail(self) -> None:
        first = record("original", "original", "duplicate")
        second = record("summary", "actual", "duplicate")
        second["segments"] = [{"segment_id": 2, "text": "Different text."}]
        report = validation.validate_records([first, second], 3, 18)
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate record_id" in error for error in report["errors"]))
        self.assertTrue(any("segment IDs" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
