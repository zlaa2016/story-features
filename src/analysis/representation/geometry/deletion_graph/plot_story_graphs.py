"""One picture per story: the directed deletion-coupling graph with its central details highlighted.

Left panel (``--layout 2d``, default): the details are placed in the plane so that strongly coupled details are close
(multidimensional scaling of ``1 / (1 + coupling)``); each arrow is a strong forward effect (deleting the tail detail changed the head), darker and thicker =
stronger.  ``--layout arc`` puts the details in story order on a line instead, with arcs.  Only the strongest ``keep`` share of pairs is drawn, the same share in every story.  ``--edges excess``
(default) ranks pairs by how much they exceed what their distance alone predicts, which shows the structure; ``--edges raw`` ranks by
absolute weight, which is dominated by neighbouring details.
Right panel: the full coupling matrix (row = deleted detail, column = detail that moved).

Two kinds of central detail are marked, both read straight from the graph:

* **key source** - the detail whose deletion moves later details the most beyond what distance predicts (highest mean outgoing
  weight over its later partners, each pair taken relative to the average pair at its lag);
* **key impacted** - the detail moved most by earlier deletions beyond what distance predicts (same, incoming).

These are centrality roles, not validated gists.  Whether they match what people keep when they recall a story is a separate test.
Outputs: ``<output-dir>/<story_key>.png``, ``story_nodes.csv`` (every node's strengths, PageRank and role) and ``story_edges.csv`` (every drawn pair: who changed whom, how much, and the two details' text).
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.deletion_graph.analyze import load_shards
from analysis.representation.geometry.deletion_graph.graph_math import (
    coupling_dissimilarity, excess_matrix, layout_2d, pagerank, strengths, thresholded, top_pairs, upper)

SOURCE_COLOR, IMPACTED_COLOR, BOTH_COLOR, PLAIN_COLOR = "#d95f02", "#1b6ca8", "#7b3294", "#9a9a9a"


def node_table(weights: np.ndarray, keep: float = 0.2, top_k: int = 3) -> pd.DataFrame:
    """Per detail: lag-adjusted mean outgoing and incoming weight (1 = as coupled as distance predicts), raw means, PageRank, role."""
    n = weights.shape[0]
    out, incoming = strengths(weights)
    raw_out, raw_in = strengths(weights, adjust=False)
    out_strength, in_strength = np.append(out, 0.0), np.insert(incoming, 0, 0.0)   # last has no later partner, first no earlier one
    ranks = pagerank(thresholded(weights, keep))
    sources = set(np.argsort(-out_strength, kind="stable")[:top_k].tolist())
    impacted = set(np.argsort(-in_strength, kind="stable")[:top_k].tolist())
    roles = ["key source + key impacted" if i in sources and i in impacted else "key source" if i in sources
             else "key impacted" if i in impacted else "" for i in range(n)]
    return pd.DataFrame({"detail_index": np.arange(n), "out_strength": out_strength, "in_strength": in_strength,
                         "out_raw": np.append(raw_out, 0.0), "in_raw": np.insert(raw_in, 0, 0.0), "pagerank": ranks, "role": roles})


def node_color(role: str) -> str:
    return BOTH_COLOR if "+" in role else SOURCE_COLOR if role == "key source" else IMPACTED_COLOR if role == "key impacted" else PLAIN_COLOR


def draw_story(weights: np.ndarray, nodes: pd.DataFrame, title: str, path: Path, keep: float = 0.2,
               texts: list[str] | None = None, edges: str = "excess", layout: str = "2d", pairs: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    n = weights.shape[0]
    ranking = excess_matrix(weights) if edges == "excess" else upper(weights)
    graph = thresholded(ranking, keep)
    peak = graph.max() if graph.max() > 0 else 1.0
    roles = nodes["role"].tolist()
    bottom = 0.36 if texts else 0.08
    figure = plt.figure(figsize=(15, 8.6 if texts else 6.6))
    axis = figure.add_axes([0.03, bottom, 0.62, 0.84 - bottom + 0.08])
    matrix = figure.add_axes([0.72, bottom, 0.25, 0.84 - bottom + 0.08])
    rows, cols = np.nonzero(graph)
    order = sorted(zip(rows, cols), key=lambda pair: graph[pair])
    total = (nodes["out_strength"] + nodes["in_strength"]).to_numpy()
    size = 60 + 320 * total / max(total.max(), 1e-12)
    if layout == "2d":
        points = layout_2d(coupling_dissimilarity(weights, edges))
        for i, j in order:
            strength = graph[i, j] / peak
            axis.add_patch(FancyArrowPatch(points[i], points[j], connectionstyle="arc3,rad=0.12", arrowstyle="-|>", mutation_scale=9 + 8 * strength,
                                           linewidth=0.5 + 3.2 * strength, color=plt.cm.plasma_r(0.15 + 0.8 * strength), alpha=0.35 + 0.55 * strength,
                                           shrinkA=6, shrinkB=7, zorder=2))
        axis.scatter(points[:, 0], points[:, 1], s=size, c=[node_color(role) for role in roles], edgecolors="black", linewidths=0.6, zorder=3)
        for i in range(n):
            axis.annotate(str(i + 1), points[i], ha="center", va="center", fontsize=7, fontweight="bold", color="white" if roles[i] else "black", zorder=4)
        pad = 0.08 * np.ptp(points, axis=0).max()
        axis.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
        axis.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
        axis.set_aspect("equal")
        axis.set_title(f"details placed so that closer = more coupled; an arrow means deleting its tail detail changed its head ({edges} ranking, darker/thicker = stronger)", fontsize=9)
    else:
        theta = np.linspace(0, np.pi, 40)
        for i, j in order:
            centre, radius = (i + j) / 2, (j - i) / 2
            color = SOURCE_COLOR if roles[i] and "source" in roles[i] else IMPACTED_COLOR if roles[j] and "impacted" in roles[j] else PLAIN_COLOR
            axis.plot(centre - radius * np.cos(theta), radius * 0.9 * np.sin(theta), color=color, linewidth=0.4 + 3.6 * graph[i, j] / peak,
                      alpha=0.25 + 0.6 * graph[i, j] / peak, solid_capstyle="round")
        axis.scatter(np.arange(n), np.zeros(n), s=size, c=[node_color(role) for role in roles], edgecolors="black", linewidths=0.6, zorder=3)
        for i, role in enumerate(roles):
            axis.annotate(str(i + 1), (i, 0), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7, fontweight="bold" if role else "normal")
        height = max([0.9 * (j - i) / 2 for i, j in zip(rows, cols)] + [1.0])
        axis.set_xlim(-1, n)
        axis.set_ylim(-0.12 * height - 0.5, height * 1.08)
        axis.set_title(f"forward dependencies in story order ({edges} ranking; an arc means deleting the left detail moved the right one)", fontsize=9)
    axis.axis("off")
    from matplotlib.lines import Line2D
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, markeredgecolor="black", label=label)
                         for c, label in [(SOURCE_COLOR, "key source"), (IMPACTED_COLOR, "key impacted"), (BOTH_COLOR, "both"), (PLAIN_COLOR, "other")]],
                loc="upper right", fontsize=8, frameon=False)

    shown = upper(weights)
    image = matrix.imshow(np.where(np.tril(np.ones((n, n), dtype=bool)), np.nan, shown), cmap="viridis", aspect="equal")
    matrix.set_xlabel("detail that moved", fontsize=8)
    matrix.set_ylabel("deleted detail", fontsize=8)
    ticks = list(range(0, n, max(1, n // 8)))
    matrix.set_xticks(ticks, [str(t + 1) for t in ticks], fontsize=7)
    matrix.set_yticks(ticks, [str(t + 1) for t in ticks], fontsize=7)
    for i, role in enumerate(roles):
        if role:
            matrix.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1, fill=False, edgecolor=node_color(role), linewidth=1.0))
    figure.colorbar(image, ax=matrix, fraction=0.046, pad=0.03)
    figure.suptitle(title, fontsize=11)

    if texts:
        strongest = top_pairs(weights, pairs, edges)
        lines = [f"Strongest effects ({'beyond what distance predicts' if edges == 'excess' else 'raw weight'}): deleting A changed B"]
        for i, j, weight, value in strongest:
            lines.append(f"  {i + 1:>2} -> {j + 1:<2} (x{value:.1f})  A: " + textwrap.shorten(texts[i], 70, placeholder="...") + "   B: " + textwrap.shorten(texts[j], 70, placeholder="..."))
        marked = [i for i, role in enumerate(roles) if role]
        lines.append("Marked details (centrality roles, not validated gists):")
        lines.extend(f"  {i + 1:>2} [{roles[i]}]  " + textwrap.shorten(texts[i], 140, placeholder="...") for i in marked)
        figure.text(0.03, 0.015, "\n".join(lines), fontsize=6.8, va="bottom", family="monospace")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def edge_table(weights: np.ndarray, story_key: str, keep: float, edges: str, texts: list[str] | None) -> pd.DataFrame:
    """Every kept forward pair of one story: who changed whom, by how much, and what the two details say."""
    n = weights.shape[0]
    ranking = excess_matrix(weights) if edges == "excess" else upper(weights)
    graph = thresholded(ranking, keep)
    rows, cols = np.nonzero(graph)
    frame = pd.DataFrame({"story_key": story_key, "source": rows + 1, "target": cols + 1, "lag": cols - rows,
                          "weight": upper(weights)[rows, cols], "excess": excess_matrix(weights)[rows, cols]})
    if texts and len(texts) == n:
        frame["source_text"] = [texts[i] for i in rows]
        frame["target_text"] = [texts[j] for j in cols]
    return frame.sort_values("excess" if edges == "excess" else "weight", ascending=False)


def detail_texts(cohorts: list[Path], wanted: set[str]) -> dict[str, list[str]]:
    from shared.io import read_jsonl

    found: dict[str, list[str]] = {}
    for path in cohorts:
        for record in read_jsonl(path):
            key = record.get("story_key")
            if key in wanted and key not in found and record.get("record_type") == "original":
                found[key] = [str(segment.get("text", "")) for segment in record.get("segments") or []]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="folder with the deletion_graph_*.npz shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=Path, default=None, help="batch folder (subset_stories.csv): picks the most viewed stories and adds each story's age range to its title")
    parser.add_argument("--stories", default="", help="comma-separated story keys to draw (default: the --n-stories most viewed in the batch, or the first ones saved)")
    parser.add_argument("--n-stories", type=int, default=12)
    parser.add_argument("--min-details", type=int, default=8, help="skip stories with fewer details when choosing automatically")
    parser.add_argument("--cohort", type=Path, nargs="*", default=[], help="cohort jsonl files: adds the text of the marked details under each picture")
    parser.add_argument("--view", choices=["mean", "last"], default="mean")
    parser.add_argument("--distance", choices=["cos", "rel"], default="cos")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--keep", type=float, default=0.2)
    parser.add_argument("--layout", choices=["2d", "arc"], default="2d", help="2d: closer = more coupled; arc: details in story order on a line")
    parser.add_argument("--pairs", type=int, default=6, help="strongest effects listed under each picture")
    parser.add_argument("--edges", choices=["excess", "raw"], default="excess", help="rank the drawn arcs by weight beyond what distance predicts, or by raw weight")
    parser.add_argument("--top-k", type=int, default=3, help="how many key sources and key impacted to mark per story")
    args = parser.parse_args()

    metadata, matrices, layer = load_shards(args.input_dir, args.view, args.distance, args.layer)
    metadata = metadata.drop_duplicates("story_key").set_index("story_key")
    info = None
    if args.batch:
        info = pd.read_csv(args.batch / "subset_stories.csv", low_memory=False).drop_duplicates("story_key").set_index("story_key")
    if args.stories:
        chosen = [key for key in args.stories.split(",") if key in metadata.index]
    else:
        pool = metadata[metadata["n_details"] >= args.min_details]
        if info is not None and "view_count" in info.columns:
            views = info["view_count"].reindex(pool.index)
            pool = pool.assign(view_count=views).sort_values("view_count", ascending=False, kind="stable")
        chosen = pool.index.tolist()[:args.n_stories]
    texts = detail_texts(args.cohort, set(chosen)) if args.cohort else {}

    tables, edge_frames = [], []
    for key in chosen:
        weights = matrices[metadata.loc[key, "key"]]
        nodes = node_table(weights, args.keep, args.top_k)
        label = f"{key}  ({int(metadata.loc[key, 'n_details'])} details"
        if info is not None and key in info.index and pd.notna(info.loc[key, "range_lower"]):
            label += f", stated age {int(info.loc[key, 'range_lower'])}-{int(info.loc[key, 'range_upper'])}"
        draw_story(weights, nodes, label + f", layer {layer}, {args.view}/{args.distance})", args.output_dir / f"{key}.png", args.keep,
                   texts.get(key) if texts.get(key) and len(texts[key]) == len(nodes) else None, args.edges, args.layout, args.pairs)
        edge_frames.append(edge_table(weights, key, args.keep, args.edges, texts.get(key)))
        tables.append(nodes.assign(story_key=key, detail_text=texts.get(key, [""] * len(nodes)) if len(texts.get(key, [])) == len(nodes) else ""))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(args.output_dir / "story_nodes.csv", index=False)
        pd.concat(edge_frames, ignore_index=True).to_csv(args.output_dir / "story_edges.csv", index=False)
    print(f"drew {len(chosen)} stories into {args.output_dir}")


if __name__ == "__main__":
    main()
