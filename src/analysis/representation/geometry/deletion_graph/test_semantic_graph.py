from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from analysis.representation.geometry.deletion_graph import semantic_graph


def decaying(n: int, rng: np.random.Generator) -> np.ndarray:
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = np.exp(-0.3 * (j - i)) * (1 + 0.5 * rng.random())
    return matrix


class AgreementTests(unittest.TestCase):
    def test_semantic_matrix_is_forward_only_and_non_negative(self) -> None:
        matrix = semantic_graph.semantic_matrix(np.random.default_rng(0).normal(size=(7, 12)))
        self.assertEqual(matrix.shape, (7, 7))
        self.assertTrue(np.allclose(np.tril(matrix), 0))
        self.assertGreaterEqual(matrix.min(), 0)

    def test_identical_graphs_agree_completely(self) -> None:
        matrix = decaying(12, np.random.default_rng(1))
        result = semantic_graph.edge_agreement(matrix, matrix)
        self.assertAlmostEqual(result["rho_raw"], 1.0)
        self.assertAlmostEqual(result["rho_excess"], 1.0)
        self.assertAlmostEqual(result["top_overlap"], 1.0)
        self.assertEqual(result["source_overlap"], 1.0)
        self.assertEqual(result["impacted_overlap"], 1.0)

    def test_shared_distance_decay_is_not_counted_as_agreement_after_lag_adjustment(self) -> None:
        rng = np.random.default_rng(2)
        rho_raw, rho_excess = [], []
        for _ in range(30):
            result = semantic_graph.edge_agreement(decaying(20, rng), decaying(20, rng))
            rho_raw.append(result["rho_raw"])
            rho_excess.append(result["rho_excess"])
        self.assertGreater(np.mean(rho_raw), 0.5)  # both fade with distance
        self.assertLess(abs(np.mean(rho_excess)), 0.15)  # nothing else is shared

    def test_chance_overlap_is_below_one_and_positive(self) -> None:
        result = semantic_graph.edge_agreement(decaying(15, np.random.default_rng(3)), decaying(15, np.random.default_rng(4)))
        self.assertTrue(0 < result["top_overlap_chance"] < 1)
        self.assertAlmostEqual(result["chance_node_overlap"], 3 / 14)


class CrossGraphR2Tests(unittest.TestCase):
    def test_semantic_gain_is_large_when_semantic_drives_deletion_and_small_when_independent(self) -> None:
        rng = np.random.default_rng(5)
        gains_dependent, gains_independent = [], []
        for _ in range(20):
            semantic = np.triu(rng.random((20, 20)), 1)  # no distance structure: the gain is what the semantic weights explain
            dependent = semantic ** 2 * (1 + 0.05 * rng.random(semantic.shape))
            gains_dependent.append(semantic_graph.cross_graph_r2(dependent, semantic)["r2_deletion_gain_semantic"])
            gains_independent.append(semantic_graph.cross_graph_r2(np.triu(rng.random((20, 20)), 1), semantic)["r2_deletion_gain_semantic"])
        self.assertGreater(np.mean(gains_dependent), 0.3)
        self.assertLess(np.mean(gains_independent), 0.1)

    def test_lag_alone_explains_a_pure_decay(self) -> None:
        n = 15
        pure = np.triu(np.array([[np.exp(-0.4 * abs(j - i)) for j in range(n)] for i in range(n)]), 1)
        result = semantic_graph.cross_graph_r2(pure, decaying(n, np.random.default_rng(6)))
        self.assertGreater(result["r2_deletion_lag"], 0.95)


class EndToEndTests(unittest.TestCase):
    def test_main_writes_agreement_and_comparison(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = [f"v{i}__story01" for i in range(14)]
            shard, details, rows, vectors, meta = root / "del", root / "det", [], [], []
            shard.mkdir(), details.mkdir()
            arrays = {"layers": np.array([0, 5])}
            for row, key in enumerate(keys):
                n = int(rng.integers(6, 12))
                base = rng.normal(size=(n, 8))
                weights = np.stack([decaying(n, rng) for _ in range(2)]).astype(np.float32)
                for name in ("cos_mean", "rel_mean", "cos_last", "rel_last"):
                    arrays[f"s{row:04d}_{name}"] = weights
                meta.append({"row": row, "story_key": key, "record_id": f"r{row}", "video_id": f"v{row}", "n_details": n, "token_count": 100})
                if row != 13:  # one story has no detail vectors
                    for d in range(n):
                        rows.append({"record_id": f"r{row}", "record_type": "original", "story_key": key, "video_id": f"v{row}",
                                     "source_age": 5.0, "detail_index": d})
                        vectors.append(base[d] + 0.3 * rng.normal(size=8))
            np.savez(shard / "deletion_graph_000.npz", **arrays)
            pd.DataFrame(meta).to_csv(shard / "deletion_graph_metadata_000.csv", index=False)
            np.savez(details / "isolated_detail_features_000.npz", mean_vectors=np.stack(vectors), eos_vectors=np.stack(vectors), layers=np.array([-1]))
            pd.DataFrame(rows).to_csv(details / "isolated_detail_metadata_000.csv", index=False)
            batch = root / "batch"
            batch.mkdir()
            pd.DataFrame({"story_key": keys, "range_lower": [3 + i % 5 for i in range(14)], "range_upper": [5 + i % 5 for i in range(14)],
                          "view_count": 1}).to_csv(batch / "subset_stories.csv", index=False)
            pd.DataFrame([{"age": 4 + i % 4, "story_key": k} for i, k in enumerate(keys)]).to_csv(batch / "membership.csv", index=False)
            out = root / "out"
            argv = ["x", "--input-dir", str(shard), "--detail-dir", str(details), "--batch", str(batch), "--output-dir", str(out)]
            with mock.patch.object(sys, "argv", argv):
                semantic_graph.main()
            agreement = pd.read_csv(out / "edge_agreement.csv")
            self.assertEqual(len(agreement), 13)
            self.assertEqual(pd.read_csv(out / "semantic_audit.csv")["status"].tolist(), ["no_detail_vectors"])
            for name in ["agreement_age_tests.csv", "measures_comparison.csv", "semantic_story_measures.csv", "fig_agreement_by_age.png", "run.json"]:
                self.assertTrue((out / name).exists(), name)
            self.assertIn("median_deletion", pd.read_csv(out / "measures_comparison.csv").columns)


if __name__ == "__main__":
    unittest.main()
