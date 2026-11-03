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
| `sv_participation`, `sv_top_share`, `sv_entropy_rank` | how many independent dependency "modes" the matrix has |

**Pictures** (`plot_story_graphs.py`, `plot.slurm`): one PNG per story. Left, the details are placed in the plane so that strongly coupled details are
close (multidimensional scaling of `1 / (1 + coupling)`) and each arrow is a strong forward effect: deleting its tail detail changed its head
(darker and thicker = stronger; `--layout arc` puts the details in story order on a line instead). Right, the full matrix. Under the picture, the
strongest effects with the text of both details ("deleting A changed B") and the marked details. Marked nodes are centrality roles read off the
graph, not validated gists: a **source hub** is a detail whose deletion moves later details more than distance predicts, an **integrator** is moved more than
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
