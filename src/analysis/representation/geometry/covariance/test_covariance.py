from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


analysis = importlib.import_module(
    "analysis.representation.geometry.covariance.analyze"
)


class CovarianceGeometryTests(unittest.TestCase):
    def test_identical_covariances_have_high_cka_and_small_angles(self) -> None:
        rng = np.random.default_rng(7)
        scales = np.asarray([5.0, 3.0, 1.0, 0.5, 0.25, 0.1])
        retained = rng.normal(size=(4000, 6)) * scales
        omitted = rng.normal(size=(4000, 6)) * scales
        result = analysis.compute_geometry(retained, omitted, subspace_rank=3)
        self.assertGreater(result.covariance_cka, 0.98)
        self.assertLess(result.mean_principal_angle_degrees, 25.0)

    def test_grouped_decoder_recovers_behavioral_signal(self) -> None:
        rng = np.random.default_rng(13)
        stories = np.repeat([f"story_{index}" for index in range(60)], 4)
        labels = np.tile([False, False, True, True], 60)
        values = rng.normal(size=(len(labels), 8))
        values[:, 0] += labels.astype(float) * 2.5
        auc = analysis.grouped_decoder_auc(values, labels, stories, folds=5, seed=13)
        self.assertGreater(auc, 0.85)

    def test_disjoint_dominant_subspaces_are_detected(self) -> None:
        rng = np.random.default_rng(11)
        retained = rng.normal(scale=0.1, size=(3000, 6))
        omitted = rng.normal(scale=0.1, size=(3000, 6))
        retained[:, :2] += rng.normal(scale=4.0, size=(3000, 2))
        omitted[:, 2:4] += rng.normal(scale=4.0, size=(3000, 2))
        result = analysis.compute_geometry(retained, omitted, subspace_rank=2)
        self.assertLess(result.covariance_cka, 0.10)
        self.assertGreater(result.mean_principal_angle_degrees, 80.0)
        self.assertGreater(result.generalized_log_ratio_mean_abs, 1.0)

    def test_permutation_preserves_each_story_retention_count(self) -> None:
        labels = np.asarray([True, False, False, True, True, False])
        stories = np.asarray(["a", "a", "a", "b", "b", "b"])
        shuffled = analysis.within_story_permutation(labels, stories, np.random.default_rng(3))
        for story in np.unique(stories):
            indices = stories == story
            self.assertEqual(int(labels[indices].sum()), int(shuffled[indices].sum()))

    def test_end_to_end_writes_research_outputs(self) -> None:
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deletion_dir = root / "deletions"
            output_dir = root / "analysis"
            deletion_dir.mkdir()
            rows = 80
            metadata = pd.DataFrame(
                {
                    "story_key": np.repeat([f"story_{index}" for index in range(10)], 8),
                    "detail_index": np.tile(np.arange(8), 10),
                    "deleted_token_count": rng.integers(2, 12, size=rows),
                }
            )
            metadata.to_csv(deletion_dir / "deletion_metadata_000.csv", index=False)
            values = rng.normal(size=(rows, 2, 8)).astype(np.float32)
            np.savez_compressed(
                deletion_dir / "deletion_features_000.npz",
                eos_deltas=values,
                mean_deltas=values + rng.normal(scale=0.05, size=values.shape),
                layers=np.asarray([0, 4]),
            )
            retention = metadata[["story_key", "detail_index"]].rename(columns={"detail_index": "orig_idx"})
            retention["retained_structured"] = np.tile([False, False, False, False, True, True, True, True], 10)
            retention["summary_condition"] = "actual"
            retention["target_age"] = 7
            retention["orig_centrality"] = rng.random(rows)
            retention_path = root / "retention.csv"
            retention.to_csv(retention_path, index=False)
            arguments = [
                "analyze.py",
                "--deletion-dir", str(deletion_dir),
                "--retention-csv", str(retention_path),
                "--output-dir", str(output_dir),
                "--pca-components", "4",
                "--subspace-rank", "2",
                "--minimum-per-class", "5",
                "--bootstrap-iterations", "2",
                "--permutation-iterations", "2",
            ]
            with patch.object(sys, "argv", arguments):
                analysis.main()
            geometry = pd.read_csv(output_dir / "covariance_geometry.csv")
            self.assertFalse(geometry.empty)
            self.assertIn("retention_decoder_story_cv_auc_global_pca", geometry)
            self.assertTrue((output_dir / "generalized_directions.npz").exists())
            self.assertTrue((output_dir / "detail_covariance_scores.csv.gz").exists())


if __name__ == "__main__":
    unittest.main()
