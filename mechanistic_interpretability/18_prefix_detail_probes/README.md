# Controlled prefix–detail probes

This experiment holds a target detail's token IDs fixed while changing only its preceding context.
It is separate from `02_representation_extraction`, and never overwrites the existing global,
contextual-detail, isolated-detail, or deletion shards.

## Sequences

For prefix `P`, target detail `D`, delimiter `S`, and fixed readout `Q`:

```text
prefix probe: BOS P S Q
detail probe: BOS P S D S Q
```

The default conditions are:

- `isolated`: empty `P`;
- `true`: details before a selected story boundary in their original order;
- `shuffled`: the same prefix details in random order and exactly the same token count;
- `foreign`: an exactly token-count-matched prefix from other stories.

The Python extractor supports all four by default. The SLURM launcher deliberately defaults to
`isolated,true`, which is the primary native-context retention experiment. Run shuffled and foreign
conditions on a deterministic sampled control cohort before considering a larger run.

By default a target already occurring before an all-prefix boundary is removed from that prefix.
This avoids measuring trivial repetition. Use `--no-exclude-target-from-prefix` only when repetition
is the intended manipulation.

## Saved measurements

Every probe row saves layerwise first-token, last-token, and mean-pooled representations of `D`,
plus the hidden state of the identical endpoint of `Q` before and after `D`. It also saves:

- teacher-forced NLL for each token of `D`;
- exact tokenwise Jensen–Shannon divergence between contextual and isolated output distributions;
- exact JS divergence between next-token distributions at `Q` before and after `D`;
- entropy change at that readout;
- token and prefix lengths, positions, condition, permutation, and truncation audit information.

Vocabulary-sized distributions are reduced online and are not saved. `--save-token-states` stores
the full ragged target-token hidden-state matrix at every selected layer and is intentionally opt-in.

## Analyses

`analyze.py` creates:

1. mean/last/first contextualization shifts relative to the identical isolated detail;
2. fixed-readout hidden-state change and output-distribution change;
3. a TOST equivalence control for mean pooling versus the final detail token;
4. actual-summary, two-row-per-summary mixed models for retained and omitted details:
   `metric ~ source_age * retention_status + detail_length + prefix_length + (1 | story)`;
5. explicit retained and omitted age slopes, their slope difference, their age-eight difference,
   Holm-adjusted p-values, and paired Cohen's dz;
6. a retention × isolated-semantic-centrality mixed model;
7. for `prefix-mode=all`, story-clustered prefix-trajectory models and plots.

The semantic-centrality split is computed only from isolated mean-pooled detail representations in
the final selected layer, preventing preceding-context length from defining centrality.
