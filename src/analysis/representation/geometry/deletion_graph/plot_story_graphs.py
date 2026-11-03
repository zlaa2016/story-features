"""One picture per story: the directed deletion-coupling graph with its central details highlighted.

Left panel: details in story order along a line; each arc is a strong forward edge (deleting the earlier detail moved the later
one), thicker = stronger.  Only the strongest ``keep`` share of pairs is drawn, the same share in every story.  ``--edges excess``
(default) ranks pairs by how much they exceed what their distance alone predicts, which shows the structure; ``--edges raw`` ranks by
absolute weight, which is dominated by neighbouring details.
Right panel: the full coupling matrix (row = deleted detail, column = detail that moved).

Two kinds of central detail are marked, both read straight from the graph:

* **source hub** - the detail whose deletion moves later details the most beyond what distance predicts (highest mean outgoing
  weight over its later partners, each pair taken relative to the average pair at its lag);
* **integrator** - the detail moved most by earlier deletions beyond what distance predicts (same, incoming).

These are centrality roles, not validated gists.  Whether they match what people keep when they recall a story is a separate test.
Outputs: ``<output-dir>/<story_key>.png`` and ``story_nodes.csv`` (every node's strengths, PageRank and role).
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.representation.geometry.deletion_graph.analyze import load_shards
from analysis.representation.geometry.deletion_graph.graph_math import excess_matrix, pagerank, strengths, thresholded, upper

HUB_COLOR, INTEGRATOR_COLOR, BOTH_COLOR, PLAIN_COLOR = "#d95f02", "#1b6ca8", "#7b3294", "#9a9a9a"


def node_table(weights: np.ndarray, keep: float = 0.2, top_k: int = 3) -> pd.DataFrame:
    """Per detail: lag-adjusted mean outgoing and incoming weight (1 = as coupled as distance predicts), raw means, PageRank, role."""
    n = weights.shape[0]
    out, incoming = strengths(weights)
    raw_out, raw_in = strengths(weights, adjust=False)
    out_strength, in_strength = np.append(out, 0.0), np.insert(incoming, 0, 0.0)   # last has no later partner, first no earlier one
    ranks = pagerank(thresholded(weights, keep))
    hubs = set(np.argsort(-out_strength, kind="stable")[:top_k].tolist())
    integrators = set(np.argsort(-in_strength, kind="stable")[:top_k].tolist())
    roles = ["source hub + integrator" if i in hubs and i in integrators else "source hub" if i in hubs
             else "integrator" if i in integrators else "" for i in range(n)]
    return pd.DataFrame({"detail_index": np.arange(n), "out_strength": out_strength, "in_strength": in_strength,
                         "out_raw": np.append(raw_out, 0.0), "in_raw": np.insert(raw_in, 0, 0.0), "pagerank": ranks, "role": roles})


def node_color(role: str) -> str:
    return BOTH_COLOR if "+" in role else HUB_COLOR if role == "source hub" else INTEGRATOR_COLOR if role == "integrator" else PLAIN_COLOR


def draw_story(weights: np.ndarray, nodes: pd.DataFrame, title: str, path: Path, keep: float = 0.2,
               texts: list[str] | None = None, edges: str = "excess") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = weights.shape[0]
    graph = thresholded(excess_matrix(weights) if edges == "excess" else weights, keep)
    peak = graph.max() if graph.max() > 0 else 1.0
    roles = nodes["role"].tolist()
    figure = plt.figure(figsize=(15, 6.2 if texts is None else 7.4))
    arcs = figure.add_axes([0.04, 0.30 if texts else 0.12, 0.60, 0.62])
    matrix = figure.add_axes([0.70, 0.30 if texts else 0.12, 0.27, 0.62])

    theta = np.linspace(0, np.pi, 40)
    rows, cols = np.nonzero(graph)
    for i, j in sorted(zip(rows, cols), key=lambda pair: graph[pair]):
        centre, radius = (i + j) / 2, (j - i) / 2
        color = HUB_COLOR if roles[i] and "hub" in roles[i] else INTEGRATOR_COLOR if roles[j] and "integrator" in roles[j] else PLAIN_COLOR
        arcs.plot(centre - radius * np.cos(theta), radius * 0.9 * np.sin(theta), color=color,
                  linewidth=0.4 + 3.6 * graph[i, j] / peak, alpha=0.25 + 0.6 * graph[i, j] / peak, solid_capstyle="round")
    total = (nodes["out_strength"] + nodes["in_strength"]).to_numpy()
    size = 40 + 340 * total / max(total.max(), 1e-12)
    arcs.scatter(np.arange(n), np.zeros(n), s=size, c=[node_color(role) for role in roles], edgecolors="black", linewidths=0.6, zorder=3)
    for i, role in enumerate(roles):
        arcs.annotate(str(i + 1), (i, 0), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7,
                      fontweight="bold" if role else "normal")
    arcs.set_xlim(-1, n)
    height = max([0.9 * (j - i) / 2 for i, j in zip(rows, cols)] + [1.0])
    arcs.set_ylim(-0.12 * height - 0.5, height * 1.08)
    arcs.axis("off")
    arcs.set_title(f"forward dependencies ({edges} ranking; an arc means deleting the left detail moved the right one; node size = coupling beyond distance)", fontsize=9)
    from matplotlib.lines import Line2D
    arcs.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, markeredgecolor="black", label=label)
                         for c, label in [(HUB_COLOR, "source hub"), (INTEGRATOR_COLOR, "integrator"), (BOTH_COLOR, "both"), (PLAIN_COLOR, "other")]],
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
        marked = [i for i, role in enumerate(roles) if role]
        lines = [f"{i + 1}  [{roles[i]}]  " + textwrap.shorten(texts[i], 150, placeholder="...") for i in marked]
        figure.text(0.04, 0.02, "\n".join(textwrap.fill(line, 165, subsequent_indent="      ") for line in lines), fontsize=7, va="bottom", family="monospace")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


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
    parser.add_argument("--edges", choices=["excess", "raw"], default="excess", help="rank the drawn arcs by weight beyond what distance predicts, or by raw weight")
    parser.add_argument("--top-k", type=int, default=3, help="how many source hubs and integrators to mark per story")
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

    tables = []
    for key in chosen:
        weights = matrices[metadata.loc[key, "key"]]
        nodes = node_table(weights, args.keep, args.top_k)
        label = f"{key}  ({int(metadata.loc[key, 'n_details'])} details"
        if info is not None and key in info.index and pd.notna(info.loc[key, "range_lower"]):
            label += f", stated age {int(info.loc[key, 'range_lower'])}-{int(info.loc[key, 'range_upper'])}"
        draw_story(weights, nodes, label + f", layer {layer}, {args.view}/{args.distance})", args.output_dir / f"{key}.png", args.keep,
                   texts.get(key) if texts.get(key) and len(texts[key]) == len(nodes) else None, args.edges)
        tables.append(nodes.assign(story_key=key, detail_text=texts.get(key, [""] * len(nodes)) if len(texts.get(key, [])) == len(nodes) else ""))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(args.output_dir / "story_nodes.csv", index=False)
    print(f"drew {len(chosen)} stories into {args.output_dir}")


if __name__ == "__main__":
    main()
