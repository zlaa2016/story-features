from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared.research import (
    assign_story_split,
    benjamini_hochberg,
    load_split,
    save_story_split,
    select_work_shard,
)


class ResearchUtilitiesTests(unittest.TestCase):
    def test_story_split_is_deterministic_and_disjoint(self) -> None:
        stories = [f"story-{index}" for index in range(50)]
        first = assign_story_split(stories, seed=17, train_fraction=0.8)
        second = assign_story_split(reversed(stories), seed=17, train_fraction=0.8)
        self.assertEqual(
            first.sort_values("story_key")["split"].tolist(),
            second.sort_values("story_key")["split"].tolist(),
        )
        self.assertEqual(set(first["split"]), {"train", "evaluation"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            save_story_split(path, first)
            loaded = load_split(path)
            self.assertEqual(len(loaded), len(stories))

    def test_benjamini_hochberg_is_monotonic_in_rank(self) -> None:
        p_values = np.asarray([0.01, 0.04, 0.03, np.nan])
        adjusted = benjamini_hochberg(p_values)
        self.assertTrue(np.isnan(adjusted[-1]))
        order = np.argsort(p_values[:3])
        self.assertTrue(np.all(np.diff(adjusted[:3][order]) >= -1e-12))
        self.assertTrue(np.all(adjusted[:3] >= p_values[:3]))

    def test_strided_work_shards_cover_rows_exactly_once(self) -> None:
        rows = list(range(103))
        shards = [select_work_shard(rows, shard_index=index, num_shards=7) for index in range(7)]
        flattened = [value for shard in shards for value in shard]
        self.assertEqual(sorted(flattened), rows)
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
