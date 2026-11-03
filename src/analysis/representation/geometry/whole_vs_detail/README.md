# Whole-story vector against its own details, by age (no PCA)

Question: how much of a story's whole-story vector do its own detail vectors account for, and does that change with age?

Vectors are scaled to unit length and centred on their corpus mean (whole-story vectors and detail vectors separately), so every
measure describes how a story *differs* from the average story, not the direction all text shares. All measures are per story, so
they do not depend on how many stories an age has (unlike "PCs needed", which scales with the number of stories).

| measure | reads as |
|---|---|
| `cos_mean`, `cos_excess` | cosine between the whole-story vector and the mean detail; `cos_excess` subtracts the mean cosine to other stories' details |
| `cos_detail` | mean cosine between the whole-story vector and each single detail |
| `r2_span_excess` | share of the whole-story vector inside the span of the story's details, above chance (rank / dimension) |
| `auc_identification` | chance that a story's own mean detail is closer to its whole-story vector than another story's is (0.5 = details say nothing specific) |
| `prefix_auc`, `order_gain` | cosine to the whole along the story's own order (mean of the first m details), and its excess over shuffled orders of the same details: positive means the real order approaches the whole faster than a random order |

`fig_prefix_by_age.png` shows the order effect along the story (x = age, y = share of details read, colour = real minus shuffled cosine).
`age_tests.csv` gives the Spearman correlation with age, raw and with the number of details partialled out, for the midpoint of the stated
range (one row per story) and for batch membership (a story counts at every age it covers, so ages are not independent samples).

Isolated detail vectors ignore order, so `order_gain` here comes only from which details come first, not from context. Use
`DETAIL_VIEW=contextual` to read details inside the story. A high overlap can also come from a story whose details simply repeat one
theme; compare with the number of details and with `auc_identification`.

```bash
MODEL_KEY=gpt_oss_20b BATCH=batch_top400_ages2-12 DETAIL_VIEW=isolated sbatch --account=torch_pr_494_general hpc/analysis/representation/geometry/whole_vs_detail/run.slurm
```
