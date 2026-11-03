from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.deletion_graph import analyze, graph_math


def chain(n: int, strength: float = 1.0) -> np.ndarray:
    """Each detail only moves the next one."""
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i, i + 1] = strength
    return matrix


def star(n: int) -> np.ndarray:
    """Detail 0 moves every later detail; nothing else moves anything."""
    matrix = np.zeros((n, n))
    matrix[0, 1:] = 1.0
    return matrix


def decaying(n: int, rate: float = 0.6) -> np.ndarray:
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = rate ** (j - i)
    return matrix


class GraphMathTests(unittest.TestCase):
    def test_gini_extremes(self) -> None:
        self.assertAlmostEqual(graph_math.gini(np.ones(10)), 0.0, places=6)
        self.assertGreater(graph_math.gini(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])), 0.85)

    def test_lower_triangle_is_ignored(self) -> None:
        matrix = decaying(8)
        noisy = matrix + np.tril(np.full((8, 8), 5.0), -1)
        self.assertEqual(graph_math.graph_measures(matrix), graph_math.graph_measures(noisy))

    def test_scale_free_measures_do_not_change_with_scale(self) -> None:
        base = graph_math.graph_measures(decaying(10))
        scaled = graph_math.graph_measures(decaying(10) * 7.5)
        for name, value in base.items():
            if name == "mean_weight":
                self.assertAlmostEqual(scaled[name], value * 7.5, places=8)
            elif name == "efficiency":
                continue  # distance is 1 / weight, so efficiency scales with the weights
            else:
                self.assertAlmostEqual(scaled[name], value, places=6, msg=name)

    def test_decay_slope_is_negative_and_matches_rate(self) -> None:
        slope = graph_math.lag_slope(graph_math.lag_profile(decaying(12, 0.6)))
        self.assertLess(slope, -0.5)
        flat = graph_math.lag_slope(graph_math.lag_profile(np.triu(np.ones((12, 12)), 1)))
        self.assertAlmostEqual(flat, 0.0, places=8)

    def test_chain_is_deep_and_star_is_shallow(self) -> None:
        n = 10
        chain_graph = graph_math.thresholded(chain(n), 0.2)  # 0.2 of the 45 pairs = the chain's 9 edges
        star_graph = graph_math.thresholded(star(n), 0.2)
        self.assertEqual(graph_math.longest_path_edges(chain_graph), n - 1)
        self.assertEqual(graph_math.longest_path_edges(star_graph), 1)

    def test_star_is_concentrated_and_low_rank(self) -> None:
        measures = graph_math.graph_measures(star(10), keep=0.2)
        self.assertAlmostEqual(measures["sv_top_share"], 1.0, places=6)
        self.assertAlmostEqual(measures["front_loading"], 1.0, places=6)
        self.assertLess(measures["pagerank_entropy"], graph_math.graph_measures(decaying(10), keep=0.2)["pagerank_entropy"] + 1)

    def test_long_range_share_orders_local_and_global_coupling(self) -> None:
        local = graph_math.graph_measures(decaying(12, 0.3))["long_range_share"]
        spread = graph_math.graph_measures(decaying(12, 0.9))["long_range_share"]
        self.assertLess(local, spread)

    def test_strengths_average_over_available_partners(self) -> None:
        out, incoming = graph_math.strengths(np.triu(np.ones((6, 6)), 1))
        self.assertTrue(np.allclose(out, 1.0))
        self.assertTrue(np.allclose(incoming, 1.0))

    def test_components_count_isolated_blocks(self) -> None:
        graph = np.zeros((6, 6))
        graph[0, 1] = graph[1, 2] = graph[3, 4] = 1.0
        self.assertEqual(graph_math.weak_components(graph), 3)

    def test_threshold_keeps_requested_share_of_pairs(self) -> None:
        rng = np.random.default_rng(0)
        graph = graph_math.thresholded(np.triu(rng.random((10, 10)), 1), 0.2)
        self.assertEqual(int((graph > 0).sum()), int(round(0.2 * 45)))

    def test_pagerank_sums_to_one(self) -> None:
        ranks = graph_math.pagerank(graph_math.thresholded(decaying(9), 0.3))
        self.assertAlmostEqual(ranks.sum(), 1.0, places=8)
        self.assertTrue((ranks > 0).all())

    def test_too_few_details_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            graph_math.graph_measures(chain(3))


class PositionBiasTests(unittest.TestCase):
    def test_pure_distance_decay_has_no_hubs(self) -> None:
        out, incoming = graph_math.strengths(decaying(14, 0.5))
        self.assertTrue(np.allclose(out, 1.0))
        self.assertTrue(np.allclose(incoming, 1.0))
        self.assertLess(graph_math.gini(out), 1e-9)

    def test_a_planted_hub_stands_out_under_decay_wherever_it_sits(self) -> None:
        for position in (1, 6, 11):
            matrix = decaying(14, 0.5)
            matrix[position, position + 1:] *= 4.0
            out, _ = graph_math.strengths(matrix)
            self.assertEqual(int(np.argmax(out)), position)

    def test_raw_strengths_do_show_the_position_bias(self) -> None:
        out, _ = graph_math.strengths(decaying(14, 0.5), adjust=False)
        self.assertEqual(int(np.argmax(out)), 12)   # the second-to-last detail: one close partner, so the highest raw mean


class AnalyzeTests(unittest.TestCase):
    def test_partial_association_removes_a_length_effect(self) -> None:
        rng = np.random.default_rng(1)
        n_details = rng.integers(5, 40, 300)
        age = np.clip(np.round(n_details / 4 + rng.normal(0, 0.3, 300)), 2, 12)
        frame = pd.DataFrame({"m": n_details + rng.normal(0, 2, 300), "age": age, "n_details": n_details})
        result = analyze.association(frame, "m", "age")
        self.assertGreater(result["rho"], 0.8)
        self.assertLess(abs(result["partial_rho"]), 0.25)

    def test_shards_roundtrip_and_pick_layer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            matrices = np.stack([decaying(6), decaying(6) * 2])
            np.savez(folder / "deletion_graph_000.npz", layers=np.array([0, 5], dtype=np.int16), model_key=np.asarray("m"),
                     s0000_cos_mean=matrices)
            pd.DataFrame({"row": [0], "story_key": ["a__story01"], "video_id": ["a"], "token_count": [100]}).to_csv(
                folder / "deletion_graph_metadata_000.csv", index=False)
            table, loaded, layer = analyze.load_shards(folder, "mean", "cos", None)
            self.assertEqual(layer, 5)
            self.assertTrue(np.allclose(loaded["000:0"], decaying(6) * 2))
            _, first, layer_zero = analyze.load_shards(folder, "mean", "cos", 0)
            self.assertEqual(layer_zero, 0)
            self.assertTrue(np.allclose(first["000:0"], decaying(6)))
            self.assertEqual(len(table), 1)


class PlotTests(unittest.TestCase):
    def test_roles_find_the_hub_and_the_integrator(self) -> None:
        from analysis.representation.geometry.deletion_graph import plot_story_graphs

        n = 10
        hub = np.zeros((n, n))
        hub[0, 1:] = 1.0                                  # detail 0 moves everything
        self.assertEqual(plot_story_graphs.node_table(hub, 0.3, 1).loc[0, "role"], "source hub")
        sink = np.zeros((n, n))
        sink[:-1, n - 1] = 1.0                            # everything moves the last detail
        table = plot_story_graphs.node_table(sink, 0.3, 1)
        self.assertEqual(table.loc[n - 1, "role"], "integrator")
        self.assertEqual(table.loc[0, "out_strength"], 1.0 / (n - 1))
        both = np.zeros((n, n))
        both[0, 1:] = 1.0
        both[:-1, n - 1] += 3.0
        self.assertEqual(plot_story_graphs.node_table(both, 0.5, 1)["role"].tolist().count(""), n - 2)

    def test_draws_a_png_with_and_without_texts(self) -> None:
        from analysis.representation.geometry.deletion_graph import plot_story_graphs

        weights = decaying(14)
        nodes = plot_story_graphs.node_table(weights, 0.2, 2)
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / "a.png"
            labelled = Path(folder) / "b.png"
            plot_story_graphs.draw_story(weights, nodes, "t", plain)
            plot_story_graphs.draw_story(weights, nodes, "t", labelled, texts=[f"detail number {i} " * 6 for i in range(14)])
            self.assertGreater(plain.stat().st_size, 5000)
            self.assertGreater(labelled.stat().st_size, 5000)


class LayoutTests(unittest.TestCase):
    @staticmethod
    def two_blocks(n: int = 16) -> np.ndarray:
        """Details 0-7 and 8-15 form two tightly coupled groups, with only weak coupling between the groups."""
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = 1.0 if (i < n // 2) == (j < n // 2) else 0.05
        return matrix

    def test_layout_separates_coupled_groups(self) -> None:
        weights = self.two_blocks()
        points = graph_math.layout_2d(graph_math.coupling_dissimilarity(weights, "raw"))
        distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
        within = np.mean([distance[i, j] for i in range(16) for j in range(i + 1, 16) if (i < 8) == (j < 8)])
        between = np.mean([distance[i, j] for i in range(8) for j in range(8, 16)])
        self.assertLess(within * 2, between)

    def test_layout_distances_follow_the_dissimilarity(self) -> None:
        from scipy import stats

        dissimilarity = graph_math.coupling_dissimilarity(decaying(12, 0.6) + np.triu(np.full((12, 12), 0.02), 1), "raw")
        points = graph_math.layout_2d(dissimilarity)
        rows, cols = np.triu_indices(12, 1)
        rho = stats.spearmanr(np.linalg.norm(points[rows] - points[cols], axis=1), dissimilarity[rows, cols])[0]
        self.assertGreater(rho, 0.9)

    def test_layout_is_oriented_first_left_of_last(self) -> None:
        points = graph_math.layout_2d(graph_math.coupling_dissimilarity(self.two_blocks(), "raw"))
        self.assertLess(points[0, 0], points[-1, 0])
        self.assertAlmostEqual(points[-1, 1], points[0, 1], places=6)

    def test_top_pairs_are_ranked_and_forward(self) -> None:
        matrix = decaying(10, 0.5)
        matrix[2, 7] = 5.0
        best = graph_math.top_pairs(matrix, 3, "excess")[0]
        self.assertEqual((best[0], best[1]), (2, 7))
        self.assertTrue(all(i < j for i, j, _, _ in graph_math.top_pairs(matrix, 5, "raw")))

    def test_drawing_in_both_layouts_and_edge_table(self) -> None:
        from analysis.representation.geometry.deletion_graph import plot_story_graphs

        weights = self.two_blocks(12)
        nodes = plot_story_graphs.node_table(weights, 0.2, 2)
        texts = [f"Detail {i + 1} about the fox and the river and the long way home." for i in range(12)]
        with tempfile.TemporaryDirectory() as folder:
            for layout in ("2d", "arc"):
                target = Path(folder) / f"{layout}.png"
                plot_story_graphs.draw_story(weights, nodes, "t", target, texts=texts, layout=layout)
                self.assertGreater(target.stat().st_size, 5000)
        table = plot_story_graphs.edge_table(weights, "s", 0.2, "excess", texts)
        self.assertEqual(len(table), int(round(0.2 * 66)))
        self.assertTrue({"source_text", "target_text"} <= set(table.columns))
        self.assertTrue((table["source"] < table["target"]).all())


if __name__ == "__main__":
    unittest.main()
