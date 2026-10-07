# Developmental Detail-Compression Interpretability

This package keeps each HPC experiment independent while sharing one canonical cohort. Large activations and generations are written under `results/analysis/representation/_legacy_mi_outputs/`, which is intentionally ignored by Git. Lightweight aggregate reports can be copied into `mechanistic_interpretability/reports/` for Git synchronization.

The complete scientific rationale, hypotheses, experiment sequence, statistical plan, falsification criteria, and human-recall bridge are in [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md).

## Scientific scope

The primary unit is an original story detail. The experiments test whether target age changes:

1. whether a detail survives summarization;
2. the relative weight of central and peripheral details;
3. the hidden representation of the story and summary;
4. MoE expert routing;
5. causal summary behavior under activation, residual, and router interventions.

## Research logic

The package is organized as a sequence of contingent questions rather than a list of methods:

| Claim | Necessary observation | Experiment | Result that weakens the claim |
|---|---|---|---|
| Age steering changes **what information is retained** | Within the same story, target age predicts source-detail survival, centrality selection, or multi-detail compression | `13_structured_detail_alignment` | No within-story age effect after matching every summary detail and controlling age distance |
| Behavioral changes reflect a **developmental representation trajectory** | Target age is decodable across layers and matched-age summaries occupy systematically different geometry | `02`, `04`, `10` | Behavioral effects occur but representation-only probes remain at shuffled-control performance |
| The trajectory has **verbalizable intermediate content** | J-lens readouts change coherently with target age before the final layer | `11_jacobian_lens` | Readout changes are absent, final-layer-only, or unrelated to detail survival |
| Sparse features mediate **selective detail compression** | SAE features predict survival and feature interventions alter retention in the predicted direction | `12_sparse_autoencoder` | SAE prediction does not generalize by story or causal effects match random directions |
| The representation is **causally involved**, not merely correlated | Residual, router, J-lens, or SAE interventions selectively move age and retention outcomes | `05`, `06`, `07`, `11`, `12` | Interventions fail or norm-matched/random controls produce equal effects |
| Compression changes **which attributes and combinations survive** | Psycholinguistic attributes and Möbius interaction order predict preservation within story | `14`, `15` | Effects disappear across lexical, transport, and semantic preservation measures |
| An age direction induces **causal output-space geometry** | Held-out interventions produce larger retention derivatives and pullback-Fisher cost than norm-matched random directions | `16` | Age-direction effects equal random controls or fail to reproduce across model families |

Each later claim is conditional on the earlier measurement result. For example, a decodable age direction is not evidence of causal mediation until an intervention changes the corresponding detail-retention outcome.

Teacher-forcing closed-model summaries through GPT-OSS measures GPT-OSS's representation of those texts. It does not recover the hidden states used by GPT-5.4-mini or GPT-5.6-luna. Native GPT-OSS generations are therefore included for direct causal claims.

## Confirmatory safeguards

- `00_dataset_index` now fails before GPU submission when identifiers, age conditions, segment numbering, source links, or compression metadata violate the canonical schema.
- `04_layerwise_probes/story_split.csv` is a seeded, deterministic train/evaluation split shared by every learned direction, J-lens token selection, domain SAE, feature selection, and causal intervention.
- Representation transforms and feature selection are fitted inside grouped training folds. Reported prediction metrics are out-of-fold by story, with story-clustered bootstrap intervals where applicable.
- Confirmatory intervention scripts default to the complete evaluation split (`--limit 0`). Positive limits are explicit pilot/debug overrides and must not be reported as corpus-level results.
- Generated summaries store deterministic per-record seeds and achieved versus requested compression. Representation extraction emits segment-coverage audits and aborts on silent truncation or failed detail localization.
- Tabular and JSON outputs use atomic replacement, and primary analysis folders include `run_provenance.json` with arguments, package versions, and the Git commit.
- Multiple related hypothesis tests report Benjamini–Hochberg adjusted q-values. Raw p-values remain available for preregistered primary contrasts.

## Empirical grounding

The experiment design is motivated by converging results, but it does not assume that an LLM is a cognitive model of a child.

- **Narrative compression is a structured transformation.** Human verbal retellings compress event duration, and the amount of compression predicts systematic encoding-to-recall representational change in high-level association cortex ([Musz & Chen, 2022](https://www.nature.com/articles/s42003-022-03418-5)). This motivates measuring detail survival, span integration, and representation change rather than treating shortening as word deletion alone.
- **Centrality is behaviorally and neurally consequential.** Semantically and causally central events are recalled more often and evoke stronger memory-related neural signatures ([Lee & Chen, 2022](https://www.nature.com/articles/s41467-022-31965-2)). This directly motivates source-detail graph centrality and retained-versus-omitted comparisons.
- **Narrative organization develops.** Event structure constrains information accessibility less strongly in children than adults ([Ren et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S0010027721002997)); developmental differences also occur in memory reactivation and inference from ages 7–30 ([Schlichting et al., 2022](https://www.nature.com/articles/s41562-021-01206-5)). Retells from children aged 3–7 further support separately coding central and peripheral idea units ([Russell, Wang, & Cain, 2024](https://www.tandfonline.com/doi/full/10.1080/10409289.2024.2303908)). These findings motivate continuous target-age tests, not a claim that one universal age boundary exists.
- **Order and integration are distinct outcomes.** Event boundaries can trade off content recognition against temporal-order memory ([Zheng et al., 2022](https://www.nature.com/articles/s41593-022-01020-w)), while relevant remote events are selectively reactivated at narrative boundaries ([Hahamy, Dubossarsky, & Behrens, 2023](https://www.nature.com/articles/s41593-023-01331-6)). This motivates reporting chronology, inversions, local span size, and dispersed integration separately.
- **Event representations are componential and context-sensitive.** Cortico-hippocampal systems reuse person, context, schema, and event-specific representations differently ([Reagh & Ranganath, 2023](https://www.nature.com/articles/s41467-023-36805-5)); top-down task focus can shift behavioral and neural event boundaries for identical narratives ([De Soares et al., 2024](https://doi.org/10.1016/j.cub.2024.09.013)). Therefore age prompts are treated as interventions on information selection, not merely metadata labels.
- **Behavior and representation must be tested separately.** Representational alignment is not equivalent to behavioral alignment ([Sucholutsky et al., 2025](https://openreview.net/forum?id=Hiq7lUh4Yn)), and functional similarity can analytically dissociate from representational similarity ([Braun, Grant, & Saxe, 2025](https://proceedings.mlr.press/v267/braun25a.html)). This motivates the behavior-only, representation-only, combined, and causal-intervention branches.

## Data warning

The complete available `younger/actual/older` corpus is the confirmatory behavioral cohort, despite its broad achieved compression distribution. Compression ratio is therefore measured and controlled rather than used to discard most observations. Grid-steering and variable-compression corpora remain exploratory appendix analyses. For cross-model concordance, `03_native_generation` matches each available GPT story-condition word budget; fixed-ratio generation is retained only as an explicit optional control.

The condition index never silently drops invalid age conditions. It records `condition_eligible` and `condition_skip_reason` for lower-bound minus three and upper-bound plus three conditions.

## Folder order

| Folder | Hardware | Purpose |
|---|---|---|
| `00_dataset_index` | CPU | Build canonical original/summary/condition tables |
| `01_smoke_test` | 1 GPU | Verify model, hidden states, router logits, generation |
| `02_representation_extraction` | GPU array | Save per-detail pooled hidden states and router summaries |
| `03_native_generation` | GPU array | Generate open-model summaries matched to available GPT story-condition budgets |
| `04_layerwise_probes` | CPU | Grouped age, condition, and detail-survival probes |
| `05_causal_patching` | 1 GPU | Patch actual-age prompt residuals into younger/older contexts |
| `06_activation_steering` | 1 GPU | Generate under age directions and norm-matched random controls |
| `07_router_analysis` | CPU + GPU | Analyze routing and perform router-only patching |
| `08_demixing_normalization` | CPU | Exploratory appendix baseline; not a primary mechanistic claim |
| `09_directional_jacobian` | 1 GPU | Estimate directional next-token Jacobians by finite differences |
| `10_behavior_representation_dissociation` | CPU | Compare behavioral, representational, combined, and shuffled controls |
| `11_jacobian_lens` | GPU arrays | Fit and apply the corpus-averaged J-lens from Anthropic's reference implementation |
| `12_sparse_autoencoder` | GPU arrays | Train token-level TopK SAEs, interpret age features, and causally steer summaries |
| `13_structured_detail_alignment` | GPU array + CPU | Align every summary detail to 1–10 contiguous source details and test within-story age effects |
| `14_detail_characteristics` | CPU | Measure non-generative AoA, readability, affect norms, discourse cues, and centrality predictors |
| `15_mobius_detail_integration` | GPU array + CPU | Decompose singleton, pair, and triple semantic support with Möbius inversion |
| `16_causal_compression_geometry` | GPU arrays + CPU | Estimate directional retention Jacobians and output-space pullback Fisher cost |

## Confirmatory cohort located locally

The canonical inputs are not the older event-segmentation review samples. They are:

- originals: `results/features/summaries/runs/customer_age_target_or_source_2_10_preprocess/original_detail_segments_source_or_target_2_10_customer_age.jsonl`;
- primary segmented summaries: `results/features/summaries/runs/age_steering_pm3_customer_discrete_resume_gpt-5.4-mini_20260817_080237/detail_segments_current_subset_local_sentence/detail_segments_summaries.jsonl`;
- completion/backfill summaries: `results/features/summaries/runs/customer_age_target_or_source_2_10_cascade_summary_detail_min1/detail_segments_summaries.jsonl`.

Before segmentation-quality filtering, the current local files contain 3,736 detail-segmented originals and 5,751 segmented summary conditions. The confirmatory builder requires at least 95% word-count agreement between each stored text and its detail sequence, then makes the detail concatenation canonical. The resulting analytic cohort contains 3,696 originals, 274,852 original details, and 5,729 summaries: 2,782 actual, 2,789 older, and 158 younger. Provenance identifies 5,239 summaries from GPT-5.4-mini and 490 from GPT-5.6-luna. Most younger conditions are structurally unavailable because lower-bound minus three falls below age two; this is recorded rather than converted into a complete-case exclusion.

The age-grid and variable-compression corpora are exploratory appendix analyses. The primary analysis retains every available valid younger, actual, or older condition and models the resulting unbalanced within-story design.

## Open-model interpretation boundary

Passing an existing GPT-generated summary through an open model with its observed tokens supplied is teacher forcing. It measures how that open model represents the text; it does not recover GPT-5.4-mini or GPT-5.6-luna's generation-time activations. No public evidence establishes that GPT-OSS-20B is an architectural proxy for GPT-5.6-luna merely because both are from OpenAI.

The confirmatory open-model comparison therefore uses GPT-OSS-20B and Qwen3-8B as two deliberately different, tractable model families. A mechanism is called convergent only when (1) each open model behaviorally reproduces the direction of GPT detail-preservation effects and (2) held-out causal intervention effects agree across models. Native open-model generation supports model-specific causal claims; teacher-forced GPT texts support representational comparison only.

## First HPC run

From the repository root:

```bash
git pull
mkdir -p logs/mechanistic_interpretability

sbatch --account=torch_pr_494_general hpc/features/summaries/dataset_index/run.slurm
```

Inspect the dataset-index counts before downloading a model:

```bash
tail -n 80 logs/mechanistic_interpretability/mech_index-*.out
```

Then run the GPU smoke test:

```bash
MODEL_KEY=gpt_oss_20b \
sbatch --account=torch_pr_494_general \
  mechanistic_interpretability/01_smoke_test/run.slurm
```

Do not submit the full arrays until the hidden-state count and generation preview are correct. `router_available=false` is expected for dense Qwen3-8B; router analyses apply only to the MoE GPT-OSS model when its implementation exposes router logits.

## Prioritized submission

The confirmatory two-model graph submits smoke tests first and keeps every downstream job dependency-gated. It includes behavioral alignment, psycholinguistic features, Möbius integration, Qwen/GPT-OSS representations and native generations, causal geometry, J-lens, and cross-model concordance. SAE jobs are optional and off by default:

```bash
export SBATCH_ARGS="--account=torch_pr_494_general"
bash hpc/analysis/submit_developmental_compression_experiments.sh \
  | tee mechanistic_interpretability_job_ids.txt
```

For the older single-model graph, use:

```bash
export SBATCH_ARGS="--account=torch_pr_494_general"
export MODEL_KEY=gpt_oss_20b
export RETENTION_CSV="results/features/summaries/runs/age_steering_pm3_customer_discrete_resume_gpt-5.4-mini_20260817_080237/detail_segments_current_subset/analysis_q/original_detail_retention.csv"
bash hpc/analysis/submit_priority.sh | tee mechanistic_interpretability_job_ids.txt
```

`02_representation_extraction` and `03_native_generation` run concurrently after the smoke test. Probe-dependent causal experiments remain pending until extraction and probing finish.

The same submission starts the full J-lens branch. SAE jobs are disabled by default; set `RUN_SAE=1` only when sparse-feature analyses are wanted. J-lens fitting uses `Qwen/Qwen3-8B` and 100 prompts split across ten resumable jobs. These are separate from the cheaper directional-Jacobian and pooled-vector analyses.

To run either branch independently:

```bash
export SBATCH_ARGS="--account=torch_pr_494_general"
sbatch $SBATCH_ARGS hpc/analysis/representation/jacobian/jacobian_lens/setup.slurmrepresentation/jacobian/jacobian_lens/setup.slurm
# After setup succeeds:
bash hpc/analysis/representation/jacobian/jacobian_lens/submit.shrepresentation/jacobian/jacobian_lens/submit.sh
bash hpc/analysis/representation/sparse_autoencoder/submit.shrepresentation/sparse_autoencoder/submit.sh
```

To run only the highest-priority extraction:

```bash
MODEL_KEY=gpt_oss_20b RECORDS_PER_SHARD=700 \
sbatch --account=torch_pr_494_general \
  hpc/features/representation_extraction/causal_lm/detail_extraction/run.slurmrepresentation_extraction/causal_lm/detail_extraction/run.slurm
```

Change the array size in `02_representation_extraction/run.slurm` if the canonical cohort has more than 10,500 text records.

## Controls

The implemented controls include:

- optional no-age and fixed-ratio native controls, kept separate from the budget-matched confirmatory comparisons;
- structurally invalid age conditions recorded explicitly;
- achieved compression ratio and source age in probe tables;
- leave-story-out grouped validation;
- shuffled-label probes;
- norm-matched random activation directions;
- unrelated-story residual patches;
- router-only patches versus residual-stream patches;
- behavior-only, representation-only, and combined age prediction;
- behavioral baseline versus divisive-normalization detail survival.
- J-lens versus vanilla/logit-lens verbalizable readouts across age conditions;
- SAE reconstruction, sparsity, dead-feature, random-direction, and causal decoder-feature controls.
- summary-complete source alignment, null-source assignments, span-length penalties, and monotonic versus atomic transport diagnostics.

## Method references

- **Structured semantic alignment:** Null matches are essential when two texts contain unequal information; unbalanced optimal transport supplies the relevant formalism ([Arase, Bao, & Yokoi, 2023](https://aclanthology.org/2023.acl-long.219/)). Experiment `13` combines a null-capable atomic transport diagnostic with an explicitly contiguous span model because atomic OT alone cannot estimate how many source details one summary unit integrates.
- **Causal representation learning:** Interventions can identify latent structure that observational geometry alone cannot ([Ahuja et al., 2023](https://proceedings.mlr.press/v202/ahuja23a.html)). The package accordingly treats probes, CKA/RSA, and feature correlations as descriptive until residual, router, J-lens, or SAE interventions move behavior with matched controls.
- **Activation steering controls:** Steering vectors can be brittle and partly driven by spurious dataset features ([Tan et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html)). Every steering claim therefore requires held-out stories, prompt variation, norm-matched random directions, and dose-response checks.
- **J-lens:** [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) and the [Anthropic reference implementation](https://github.com/anthropics/jacobian-lens).
- **Sparse features:** [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093), [Gemma Scope](https://arxiv.org/abs/2408.05147), and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/). Feature labels remain hypotheses until interventions and random-feature controls support them.

## Report synchronization

Large `.npz` and `.jsonl` outputs should remain on scratch. Collect aggregate files for Git:

```bash
python mechanistic_interpretability/collect_reports.py
git add mechanistic_interpretability/reports
git commit -m "Add mechanistic interpretability reports"
git pull --rebase
git push
```

The experiment code itself can be pushed independently:

```bash
git add mechanistic_interpretability
git commit -m "Add developmental compression mechanistic experiments"
git pull --rebase
git push
```
