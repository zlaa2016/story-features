from __future__ import annotations

import importlib
import unittest


extraction = importlib.import_module("mechanistic_interpretability.02_representation_extraction.extract")


class RepresentationExtractionTests(unittest.TestCase):
    def test_repeated_segments_are_located_monotonically(self) -> None:
        text = "The cat ran. The dog waited. The cat ran."
        segments = [{"text": "The cat ran."}, {"text": "The dog waited."}, {"text": "The cat ran."}]
        spans = extraction.locate_segment_spans(text, segments)
        self.assertEqual(len(spans), 3)
        self.assertTrue(all(span is not None for span in spans))
        self.assertLess(spans[0][0], spans[1][0])
        self.assertLess(spans[1][0], spans[2][0])

    def test_unmatched_segment_is_explicit(self) -> None:
        spans = extraction.locate_segment_spans("Known text.", [{"text": "Missing text."}])
        self.assertEqual(spans, [None])


if __name__ == "__main__":
    unittest.main()
