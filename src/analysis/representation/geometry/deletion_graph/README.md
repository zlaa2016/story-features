# Directed deletion-coupling graph per story

Question: when one detail is removed, which *later* details change, and how does the pattern of those dependencies (the
story's directed graph) differ between stories and across ages?

**Requires** `networkx` for the analysis and picture steps (PageRank, shortest paths, components, longest path, Kamada-Kawai layout); the GPU extraction does not use it.

**Edges.** For a story with n details, one full pass gives every detail's state (mean over its tokens, and its last token) at a
few layers. Each detail i is then deleted in turn and the story re-run. Edge i -> j is how far detail j's state moved:
`cos` = 1 - cosine(full, deleted) or `rel` = ||full - deleted|| / ||full||. A causal LM cannot change an earlier state because of
a later deletion, so entries with j < i must be ~0 (`causal_floor` in the outputs: the noise floor and a check of the setup);
only the upper triangle is used.

**Measures** (`graph_math.graph_measures`), all comparable across stories of different length:
node strengths average over the partners a node can actually have (detail 1 has n-1 later partners, the last has one) and are taken relative to the average pair at the same lag (without this, details near the end look like hubs only because their few partners are close); all but
`mean_weight` are scale free; the thresholded graph (path, components, PageRank, efficiency) keeps the strongest 20% of pairs, so every
story has the same density and only its shape is compared.

| measure | reads as |
|---|---|
| `mean_weight` | overall size of the coupling |
| `lag_slope` | how fast coupling fades with distance (more negative = local) |
| `long_range_share`, `mean_lag_norm` | how much weight sits far from the deleted detail |
| `residual_cv` | selectivity beyond distance: do particular pairs matter more than their lag predicts? |
| `out_gini`, `in_gini` | hubs: few details that move many (sources), or are moved by many (targets) |
| `front_loading` | share of weight coming from the first third |
| `depth_frac`, `components_frac` | longest dependency chain / (n-1); fragmentation of the thresholded graph |
| `pagerank_entropy`, `efficiency` | spread of centrality; how short the strong paths are |
| `max_clique_frac`, `n_maximal_cliques`, `clustering`, `transitivity`, `max_core_frac` | cliques and triangles of the strong-edge graph (undirected projection): groups of details that all affect each other |
| `n_communities_frac`, `modularity` | whether the strong edges form separate communities (greedy modularity) |
| `degree_assortativity` | whether highly connected details link to other highly connected ones |
| `sources_frac`, `sinks_frac`, `feedforward_share`, `reduction_frac` | directed structure: details that only move others / are only moved; share of two-step chains that also have a direct shortcut; share of edges left after transitive reduction (1 = pure chain) |
| `*_z` | each of the structure measures against shuffles that keep the decay with distance but move which pairs are strong (0 = what distance alone gives) |
| `sv_participation`, `sv_top_share`, `sv_entropy_rank` | how many independent dependency "modes" the matrix has |

**Pictures** (`plot_story_graphs.py`, `plot.slurm`): one PNG per story. Left, the details are placed in the plane so that strongly coupled details are
close (multidimensional scaling of `1 / (1 + coupling)`) and each arrow is a strong forward effect: deleting its tail detail changed its head
(darker and thicker = stronger; `--layout arc` puts the details in story order on a line instead). Right, the full matrix. Under the picture, the
strongest effects with the text of both details ("deleting A changed B") and the marked details. Marked nodes are centrality roles read off the
graph, not validated gists: a **key source** is a detail whose deletion moves later details more than distance predicts, an **key impacted** is moved more than
distance predicts by earlier deletions; node size is total coupling beyond distance. `story_edges.csv` lists every drawn pair with both texts and
`story_nodes.csv` every node's strengths and role.

**Confounds to keep in mind.** Deleting a longer detail shifts later positions more, so run `--view`/`--distance` variants and
compare with `n_details` and `token_count` (the age tests partial out the number of details). Details that alternate in the text are
often related (cause/effect), so coupling mixes position, content and structure; a replacement control (swap detail i for a
length-matched unrelated sentence) and a shuffled-order null are the natural next controls and are not built here.

**Run** (top 100 stories per age, ages 2-12; 426 distinct stories; per-age counts in `data/processed/spectral_age/batch_top100_ages2-12/summary.json`):

```bash
MODEL_KEY=qwen3_8b BATCH=batch_top100_ages2-12 bash hpc/analysis/representation/geometry/deletion_graph/submit.sh
MODEL_KEY=qwen3_8b BATCH=batch_top100_ages2-12 sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/deletion_graph/analyze.slurm
MODEL_KEY=qwen3_8b BATCH=batch_top100_ages2-12 sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/deletion_graph/plot.slurm
```

Extraction writes `<detail_extraction>/<model>_deletion_graph_<batch>/deletion_graph_NNN.npz` (+ metadata and audit csv; the audit lists
stories skipped: not in any cohort, fewer than 4 details, over the context limit, details that could not be mapped). Analysis writes
`story_measures.csv`, `age_summary.csv`, `age_tests.csv` (Spearman with age, raw and partialling out `n_details`, for membership and
midpoint-of-range ages), `causal_check.csv` and `fig_measures_by_age.png` under `results/analysis/representation/geometry/deletion_graph/`.

## Events instead of details (`batch_events_ages2-12`)

The legacy event-segmented stories (`data/legacy/review_samples_event_segments/`, file prefix = master `story_id`) can be graphed with the same
deletion coupling: an event takes the place of a detail (`extract_deletion_graph.py --event-dir`, `UNITS=events` in the slurm scripts).
`event_batch.py` builds the batch of those stories whose reading-age range touches 2-12 (a story is in `membership.csv` once per covered age).
`compare_units.py` then sets the event graph measures against the detail graph measures of the same stories (`units_comparison.csv`).

Caveats: an event graph has about 10 nodes against tens for details, so levels differ and only ranks across stories and age trends are
comparable; `story_key` is `<video_id>__story01`, so for a video with several stories the event file may belong to another story, which
`compare_units.py` catches by requiring matching token counts; stories without a detail record in any cohort show up as `not_found` in the audit.

## Semantic graph against the deletion graph (`semantic_graph.py`, `semantic.slurm`)

The semantic graph uses the cosine of existing detail vectors (`isolated`: each detail embedded alone; `contextual`: its state inside the
story) as edge strength, on the forward pairs i < j, and goes through the same graph measures as the deletion matrices.  Nothing is
re-extracted: it needs the deletion shards and the detail-vector folders of the same model (`MODEL_DIRS` in `spectral_age/run.py`).
Per story it reports the pair-weight Spearman raw and after dividing out the lag, the overlap of the strongest pairs against chance, and how many
key source / key impacted details the two graphs share; it also puts every graph measure of both side by side.  Stories whose vector count
differs from the deletion graph's detail count are skipped and listed in `semantic_audit.csv`.
