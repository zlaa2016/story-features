from __future__ import annotations

import importlib
import unittest

import pandas as pd


analysis = importlib.import_module("mechanistic_interpretability.13_structured_detail_alignment.analyze")


class StructuredStatisticsTests(unittest.TestCase):
    def test_interaction_is_formed_before_within_story_demeaning(self) -> None:
        rows = []
        for story_index in range(30):
            source_age = 5 + story_index / 10
            story_effect = story_index / 7
            for condition, target_age, distance, matched in [
                ("younger", source_age - 3, 3, 0),
                ("actual", source_age, 0, 1),
                ("older", source_age + 3, 3, 0),
            ]:
                outcome = story_effect + 0.2 * target_age + 0.1 * distance + 0.04 * target_age * distance
                rows.append({"story_key": f"story-{story_index}", "condition": condition, "target_age": target_age, "age_distance": distance, "age_matched": matched, "outcome": outcome})
        result = analysis.clustered_fixed_effect(pd.DataFrame(rows), "outcome")
        self.assertIsNotNone(result)
        terms = {row["term"] for row in result["rows"]}
        self.assertIn("within_target_x_distance_raw", terms)
        self.assertNotIn("target_x_distance", terms)

    def test_retention_set_analysis_compares_conditions_and_baselines(self) -> None:
        rows = []
        for condition, retained in [("younger", {0, 1}), ("actual", {0, 2})]:
            for detail_index, centrality in enumerate([1.0, 0.7, 0.4, 0.1]):
                rows.append(
                    {
                        "story_key": "story",
                        "condition": condition,
                        "summary_id": 1,
                        "summary_variant": 1,
                        "orig_idx": detail_index,
                        "retained_structured": detail_index in retained,
                        "centrality_pct": centrality,
                    }
                )
        result = analysis.retention_set_analysis(pd.DataFrame(rows))
        self.assertEqual((result["row_type"] == "retention_baseline").sum(), 2)
        pair = result[result["row_type"] == "condition_pair"].iloc[0]
        self.assertAlmostEqual(pair["retention_jaccard"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
