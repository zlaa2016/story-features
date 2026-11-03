from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

extraction = importlib.import_module(
    "features.representation_extraction.causal_lm.detail_extraction.extract_deletion_graph"
)


class CouplingTests(unittest.TestCase):
    def test_identical_states_give_zero_and_orthogonal_states_give_one(self) -> None:
        rng = np.random.default_rng(0)
        full = rng.normal(size=(4, 2, 8))
        same = {1: full[[0, 2, 3]].copy()}
        cos, rel = extraction.coupling_matrices(full, same)
        self.assertTrue(np.allclose(cos, 0.0, atol=1e-9))
        self.assertTrue(np.allclose(rel, 0.0, atol=1e-9))
        orthogonal = np.zeros((1, 1, 2))
        orthogonal[0, 0] = [1.0, 0.0]
        moved = np.zeros((1, 1, 2))
        moved[0, 0] = [0.0, 1.0]
        cos, rel = extraction.coupling_matrices(orthogonal.repeat(2, axis=0), {0: moved})
        self.assertAlmostEqual(float(cos[0, 0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(rel[0, 0, 1]), float(np.sqrt(2)), places=6)

    def test_only_deleted_rows_are_filled_and_the_diagonal_stays_zero(self) -> None:
        rng = np.random.default_rng(1)
        full = rng.normal(size=(5, 3, 6))
        deleted = {2: full[[0, 1, 3, 4]] + rng.normal(scale=0.1, size=(4, 3, 6))}
        cos, _ = extraction.coupling_matrices(full, deleted)
        self.assertEqual(cos.shape, (3, 5, 5))
        self.assertTrue(np.all(cos[:, 2, 2] == 0))
        self.assertTrue(np.all(cos[:, [0, 1, 3, 4], :] == 0))
        self.assertTrue(np.all(cos[:, 2, [0, 1, 3, 4]] > 0))

    def test_wrong_row_count_is_an_error(self) -> None:
        full = np.zeros((4, 1, 3)) + 1.0
        with self.assertRaises(ValueError):
            extraction.coupling_matrices(full, {0: full[:2]})

    def test_story_records_follow_the_wanted_order_and_first_cohort_wins(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            first, second = folder / "a.jsonl", folder / "b.jsonl"
            rows_a = [{"story_key": "x", "record_type": "original", "text": "a-x"}, {"story_key": "x", "record_type": "summary", "text": "s"}]
            rows_b = [{"story_key": "x", "record_type": "original", "text": "b-x"}, {"story_key": "y", "record_type": "original", "text": "b-y"}]
            first.write_text("\n".join(json.dumps(r) for r in rows_a))
            second.write_text("\n".join(json.dumps(r) for r in rows_b))
            found = extraction.story_records([first, second], ["y", "x", "missing"])
            self.assertEqual([r["story_key"] for r in found], ["y", "x"])
            self.assertEqual(found[1]["text"], "a-x")

    def test_ordered_story_keys_rank_by_views_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stories.csv"
            pd.DataFrame({"story_key": ["a", "b", "c", "b"], "view_count": [5, 50, 20, 50]}).to_csv(path, index=False)
            self.assertEqual(extraction.ordered_story_keys(path, "story_key", 0), ["b", "c", "a"])
            self.assertEqual(extraction.ordered_story_keys(path, "story_key", 2), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
