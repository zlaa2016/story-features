"""Graph measures of one story's deletion-coupling matrix.

``W[i, j]`` (j > i) is how much deleting detail i moved detail j.  Only the upper triangle is used: a causal model
cannot move earlier details, so the lower triangle is numerical noise.  Every measure is built to be comparable across
stories with different numbers of details and different overall weight scales:

* node strengths average over the partners a node can actually have (detail 1 has n-1 later partners, detail n-1 has one) and
  are taken relative to the average pair at the same lag, so position in the story does not make a detail a hub,
* measures are scale free (shares, ratios, Gini, slopes, ranks) except ``mean_weight``,
* the thresholded graph keeps the strongest ``keep`` share of pairs, so its density is the same in every story and only
  its shape is compared.

Standard graph algorithms (PageRank, shortest paths, components, DAG longest path, Kamada-Kawai layout) come from NetworkX; the
lag adjustment, Gini, lag profile and singular-value measures are specific to this analysis and written here.
"""
from __future__ import annotations

import numpy as np

try:
    import networkx as nx
except ImportError as error:  # pragma: no cover - a clear message beats a bare ModuleNotFoundError on the cluster
    raise ImportError("graph_math needs networkx (pip install networkx); it supplies PageRank, shortest paths, components, longest path and the layout") from error

EPSILON = 1e-12


def upper(weights: np.ndarray) -> np.ndarray:
    return np.triu(np.asarray(weights, dtype=np.float64), 1)


def gini(values: np.ndarray) -> float:
    """0 when every value is equal, near 1 when one value carries everything."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size < 2 or values.sum() <= EPSILON:
        return float("nan")
    ranks = np.arange(1, values.size + 1)
    return float((2 * (ranks * values).sum() / (values.size * values.sum())) - (values.size + 1) / values.size)


def lag_profile(weights: np.ndarray) -> np.ndarray:
    """Mean weight at each lag 1 .. n-1 (index 0 is lag 1)."""
    matrix = upper(weights)
    n = matrix.shape[0]
    return np.array([np.diagonal(matrix, offset=lag).mean() for lag in range(1, n)])


def lag_slope(profile: np.ndarray) -> float:
    """Slope of log mean weight against log lag: how fast coupling fades with distance (more negative = faster)."""
    lags = np.arange(1, len(profile) + 1, dtype=np.float64)
    keep = profile > EPSILON
    if keep.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(lags[keep]), np.log(profile[keep]), 1)[0])


def residual_cv(weights: np.ndarray) -> float:
    """Selectivity beyond distance: spread of each pair's weight relative to the average pair at the same lag."""
    matrix = upper(weights)
    n = matrix.shape[0]
    ratios = []
    for lag in range(1, n - 1):
        diagonal = np.diagonal(matrix, offset=lag)
        if diagonal.size >= 2 and diagonal.mean() > EPSILON:
            ratios.extend((diagonal / diagonal.mean()).tolist())
    ratios = np.asarray(ratios)
    return float(ratios.std() / ratios.mean()) if ratios.size > 2 and ratios.mean() > EPSILON else float("nan")


def spectral(weights: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(upper(weights), compute_uv=False)
    energy = singular ** 2
    total = energy.sum()
    if total <= EPSILON:
        return {"sv_participation": float("nan"), "sv_top_share": float("nan"), "sv_entropy_rank": float("nan")}
    share = singular / singular.sum()
    nonzero = share[share > EPSILON]
    return {
        "sv_participation": float(total ** 2 / (energy ** 2).sum()),
        "sv_top_share": float(energy[0] / total),
        "sv_entropy_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
    }


def excess_matrix(weights: np.ndarray) -> np.ndarray:
    """Each pair's weight divided by the average pair at the same lag: 1 = as strong as distance alone predicts."""
    matrix = upper(weights)
    n = matrix.shape[0]
    excess = np.zeros_like(matrix)
    for lag in range(1, n):
        diagonal = np.diagonal(matrix, offset=lag)
        if diagonal.mean() > EPSILON:
            rows = np.arange(n - lag)
            excess[rows, rows + lag] = diagonal / diagonal.mean()
    return excess


def strengths(weights: np.ndarray, adjust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mean outgoing weight per source (over its later partners) and mean incoming weight per target (over its earlier partners).

    With ``adjust`` (default) each pair is first divided by the average pair at the same lag, so a detail is not a "hub" merely
    because it sits near the end (few, close partners) or the start of the story; 1 means as coupled as distance predicts.
    """
    matrix = excess_matrix(weights) if adjust else upper(weights)
    n = matrix.shape[0]
    out = np.array([matrix[i, i + 1:].mean() for i in range(n - 1)])
    incoming = np.array([matrix[:j, j].mean() for j in range(1, n)])
    return out, incoming


def thresholded(weights: np.ndarray, keep: float) -> np.ndarray:
    """Weights of the strongest ``keep`` share of pairs (ties broken by order), 0 elsewhere."""
    matrix = upper(weights)
    n = matrix.shape[0]
    rows, cols = np.triu_indices(n, 1)
    values = matrix[rows, cols]
    count = max(1, int(round(keep * values.size)))
    chosen = np.argsort(-values, kind="stable")[:count]
    graph = np.zeros_like(matrix)
    graph[rows[chosen], cols[chosen]] = values[chosen]
    return graph


def to_digraph(graph: np.ndarray) -> "nx.DiGraph":
    """Directed graph with every detail as a node (isolated ones too); edge ``weight`` = coupling, ``distance`` = 1 / coupling."""
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(graph.shape[0]))
    for i, j in zip(*np.nonzero(graph > 0)):
        digraph.add_edge(int(i), int(j), weight=float(graph[i, j]), distance=1.0 / float(graph[i, j]))
    return digraph


def longest_path_edges(graph: np.ndarray) -> int:
    """Number of edges on the longest directed path (the graph is a DAG: edges only go forward)."""
    return int(nx.dag_longest_path_length(to_digraph(graph), weight=None))


def weak_components(graph: np.ndarray) -> int:
    return int(nx.number_weakly_connected_components(to_digraph(graph)))


def pagerank(graph: np.ndarray, damping: float = 0.85) -> np.ndarray:
    ranks = nx.pagerank(to_digraph(graph), alpha=damping, weight="weight")
    return np.array([ranks[i] for i in range(graph.shape[0])])


def efficiency(graph: np.ndarray) -> float:
    """Mean inverse shortest-path distance over ordered pairs, with edge distance 1 / weight (strong coupling = short)."""
    n = graph.shape[0]
    lengths = dict(nx.all_pairs_dijkstra_path_length(to_digraph(graph), weight="distance"))
    total = sum(1.0 / length for source, reached in lengths.items() for target, length in reached.items() if target != source)
    return float(total / (n * (n - 1)))


def coupling_dissimilarity(weights: np.ndarray, edges: str = "excess") -> np.ndarray:
    """Symmetric dissimilarity for a layout: strongly coupled details are close (``1 / (1 + coupling)``, so it lies in (0, 1]).

    ``edges="excess"`` uses each pair's weight relative to the average pair at its lag (distance in the story is taken out);
    ``"raw"`` uses the weight relative to the story's mean weight (neighbouring details dominate).
    """
    matrix = excess_matrix(weights) if edges == "excess" else upper(weights) / max(upper(weights)[np.triu_indices(len(weights), 1)].mean(), EPSILON)
    coupling = matrix + matrix.T
    dissimilarity = 1.0 / (1.0 + coupling)
    np.fill_diagonal(dissimilarity, 0.0)
    return dissimilarity


def layout_2d(dissimilarity: np.ndarray) -> np.ndarray:
    """Kamada-Kawai layout (networkx): distances between points follow ``dissimilarity``.  Rotated so the first detail sits left of the last."""
    n = dissimilarity.shape[0]
    complete = nx.complete_graph(n)
    distances = {i: {j: float(dissimilarity[i, j]) for j in range(n)} for i in range(n)}
    positions = nx.kamada_kawai_layout(complete, dist=distances, scale=1.0)
    points = np.array([positions[i] for i in range(n)])
    points = points - points.mean(axis=0)
    direction = points[-1] - points[0]
    angle = -np.arctan2(direction[1], direction[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points @ rotation.T


def top_pairs(weights: np.ndarray, k: int, edges: str = "excess") -> list[tuple[int, int, float, float]]:
    """The k strongest forward pairs as (source, target, raw weight, ranking value), ranked by the excess or the raw weight."""
    ranking = excess_matrix(weights) if edges == "excess" else upper(weights)
    rows, cols = np.triu_indices(len(weights), 1)
    order = np.argsort(-ranking[rows, cols], kind="stable")[:k]
    return [(int(rows[o]), int(cols[o]), float(upper(weights)[rows[o], cols[o]]), float(ranking[rows[o], cols[o]])) for o in order]


def graph_measures(weights: np.ndarray, keep: float = 0.2) -> dict[str, float]:
    """All measures of one story's coupling matrix (n >= 4 details)."""
    matrix = upper(weights)
    n = matrix.shape[0]
    if n < 4:
        raise ValueError("a graph measure needs at least 4 details")
    rows, cols = np.triu_indices(n, 1)
    values = matrix[rows, cols]
    total = values.sum()
    lags = (cols - rows).astype(np.float64)
    out, incoming = strengths(matrix)
    profile = lag_profile(matrix)
    graph = thresholded(matrix, keep)
    ranks = pagerank(graph)
    entropy = float(-(ranks * np.log(np.maximum(ranks, EPSILON))).sum() / np.log(n))
    measures = {
        "n_details": float(n),
        "mean_weight": float(values.mean()),
        "lag_slope": lag_slope(profile),
        "long_range_share": float(values[lags >= np.ceil(n / 2)].sum() / total) if total > EPSILON else float("nan"),
        "mean_lag_norm": float((values * lags).sum() / total / (n - 1)) if total > EPSILON else float("nan"),
        "residual_cv": residual_cv(matrix),
        "out_gini": gini(out),
        "in_gini": gini(incoming),
        "front_loading": float(matrix[: max(1, n // 3)].sum() / total) if total > EPSILON else float("nan"),
        "depth_frac": longest_path_edges(graph) / (n - 1),
        "components_frac": weak_components(graph) / n,
        "pagerank_entropy": entropy,
        "efficiency": efficiency(graph),
    }
    measures.update(spectral(matrix))
    return measures


MEASURE_LABELS = {
    "mean_weight": "mean edge weight (scale of coupling)",
    "lag_slope": "lag decay slope (log-log, more negative = local)",
    "long_range_share": "share of weight at lag >= n/2",
    "mean_lag_norm": "weight-averaged lag / (n-1)",
    "residual_cv": "selectivity beyond lag (CV)",
    "out_gini": "hub concentration of sources (Gini)",
    "in_gini": "concentration of targets (Gini)",
    "front_loading": "share of weight from the first third",
    "depth_frac": "longest path / (n-1)",
    "components_frac": "weak components / n",
    "pagerank_entropy": "PageRank entropy (1 = no hubs)",
    "efficiency": "global efficiency",
    "sv_participation": "singular-value participation ratio",
    "sv_top_share": "top singular-value share",
    "sv_entropy_rank": "singular-value entropy rank",
}
