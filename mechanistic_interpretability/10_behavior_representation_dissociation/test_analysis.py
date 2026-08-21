from __future__ import annotations

import importlib
import unittest

import numpy as np


analysis = importlib.import_module("mechanistic_interpretability.10_behavior_representation_dissociation.analyze")


class DissociationAnalysisTests(unittest.TestCase):
    def test_shuffle_preserves_each_story_label_multiset(self) -> None:
        labels = np.asarray(["younger", "actual", "older"] * 4)
        groups = np.repeat(["a", "b", "c", "d"], 3)
        shuffled = analysis.shuffle_within_story(labels, groups, np.random.default_rng(8))
        for story in np.unique(groups):
            rows = groups == story
            self.assertEqual(sorted(labels[rows]), sorted(shuffled[rows]))


if __name__ == "__main__":
    unittest.main()
