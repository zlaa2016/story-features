from __future__ import annotations

import unittest

import numpy as np

from analysis.representation.geometry.deletion_graph import graph_math


def rank_one(n: int) -> np.ndarray:
    v = np.linspace(1.0, 2.0, n)
    return np.triu(np.outer(v, v), 1)


def noise(n: int, seed: int) -> np.ndarray:
    return np.triu(np.random.default_rng(seed).random((n, n)), 1)


class CompressionTests(unittest.TestCase):
    def test_low_rank_matrix_leaves_less_energy_than_noise(self) -> None:
        n = 20
        structured = graph_math.graph_measures(rank_one(n))
        random = graph_math.graph_measures(noise(n, 0))
        self.assertLess(structured["lowrank_resid3"], random["lowrank_resid3"])
        self.assertLessEqual(structured["sv_rank90_frac"], random["sv_rank90_frac"])
        self.assertTrue(0 <= structured["lowrank_resid2"] <= structured["lowrank_resid2"] + 1e-12)

    def test_edges_set_by_lag_cost_fewer_bits_than_random_edges(self) -> None:
        n = 30
        lag_driven = np.triu(np.array([[np.exp(-0.5 * abs(j - i)) for j in range(n)] for i in range(n)]), 1)
        bits_lag = graph_math.adjacency_bits(graph_math.thresholded(lag_driven, 0.2))["adj_bits_lag"]
        bits_random = np.mean([graph_math.adjacency_bits(graph_math.thresholded(noise(n, s), 0.2))["adj_bits_lag"] for s in range(10)])
        self.assertLess(bits_lag, bits_random)
        self.assertLess(bits_lag, graph_math.binary_entropy_bits(np.array(0.2)).item())

    def test_binary_entropy(self) -> None:
        self.assertAlmostEqual(float(graph_math.binary_entropy_bits(np.array(0.5))), 1.0)
        self.assertEqual(float(graph_math.binary_entropy_bits(np.array(0.0))), 0.0)

    def test_star_is_concentrated_and_a_chain_is_not(self) -> None:
        n = 12
        star = np.zeros((n, n))
        star[0, 1:] = 1.0
        chain = np.zeros((n, n))
        for i in range(n - 1):
            chain[i, i + 1] = 1.0
        s = graph_math.compression_measures(star, star)
        c = graph_math.compression_measures(chain, chain)
        self.assertLess(s["degree_share_entropy"], c["degree_share_entropy"])
        self.assertAlmostEqual(c["treewidth_frac"], 1 / (n - 1))  # a path has treewidth 1
        self.assertGreater(s["level_width_frac"], c["level_width_frac"])  # a star has one wide level
        self.assertAlmostEqual(c["level_width_frac"], 1 / n)

    def test_new_measures_are_in_graph_measures_and_labelled(self) -> None:
        measures = graph_math.graph_measures(noise(15, 3))
        for name in ["lowrank_resid2", "lowrank_resid3", "sv_rank90_frac", "degree_share_entropy", "treewidth_frac",
                     "level_width_frac", "adj_bits_lag", "adj_bits_row"]:
            self.assertIn(name, measures)
            self.assertIn(name, graph_math.MEASURE_LABELS)


if __name__ == "__main__":
    unittest.main()
