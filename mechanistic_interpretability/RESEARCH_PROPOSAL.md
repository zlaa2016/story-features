# Developmental Information Compression in Narrative Summarization

## A behavioral and mechanistic study of what language models preserve, integrate, and discard for different target ages

## Project summary

This project asks whether age-conditioned narrative summarization follows a coherent developmental trajectory at two distinct levels: **observable information selection** and **internal model representation**. The behavioral level measures which concrete story details survive, how several source details are integrated into a single summary detail, whether central information is favored over peripheral information, and whether narrative order is preserved. The mechanistic level measures where target age and detail survival are represented across model layers, whether those representations are sparse or distributed, and whether intervening on them causally changes what the model retains.

The central scientific problem is a possible dissociation: a model can generate summaries that look appropriately adapted for different ages without organizing those summaries through a corresponding developmental representation trajectory. Conversely, internal target-age information may be strongly decodable while having little causal influence on detail selection. The study therefore does not infer mechanism from output differences or from decodability alone. It progresses from behavioral measurement, to representational analysis, to controlled intervention.

The primary contribution is a **summary-complete account of narrative compression**. Every summary detail is aligned to one or more source details, with explicit null matches for unsupported content. This permits three separable measurements: whether a source detail survives, how much source information a summary detail compresses, and whether the relative weight of central versus peripheral details changes with target age. These measurements are then linked to layerwise hidden states, expert routing, Jacobian-lens readouts, and sparse-autoencoder features in open-weight language models.

## Central research question

When a language model summarizes the same story for younger, matched-age, and older audiences under a within-story matched information budget, do changes in **what the model says** correspond to changes in **how the model internally represents and causally selects narrative information**?

## Specific aims

### Aim 1: Measure developmental changes in narrative compression

Determine whether target age changes:

1. source-detail survival;
2. selection of central versus peripheral details;
3. the number of source details integrated into each summary detail;
4. fusion of consecutive versus dispersed information;
5. unsupported or weakly grounded summary content; and
6. preservation of chronological order.

This aim establishes whether there is a reliable behavioral phenomenon before interpreting internal activations.

### Aim 2: Identify an internal developmental representation trajectory

Test whether target age and detail survival are encoded across layers, whether matched-age summaries occupy a distinct representational geometry, and whether mixture-of-experts routing changes with age condition. Compare behavioral-only, representation-only, and combined prediction under leave-story-out validation.

This aim asks whether representational differences add information beyond visible output properties such as summary length, number of details, and achieved compression.

### Aim 3: Test whether the trajectory causally controls information selection

Intervene on residual-stream directions, expert routing, Jacobian-lens coordinates, and sparse-autoencoder features. Measure whether these interventions change detail survival and compression in the predicted developmental direction, relative to norm-matched random, unrelated-story, and no-intervention controls.

This aim distinguishes causal mediation from correlation or probe accessibility.

### Aim 4: Build a transferable measurement framework for human narrative recall

Apply the same source-detail survival, centrality, compression, and order measures to age-indexed human recall transcripts when they become available. Test whether model-derived features predict which details people recall in a data-scarce developmental sample.

This aim does not claim that the model simulates children. It tests whether model-based measurement and discovery can augment behavioral work when participant data are limited.

## Theoretical motivation

Human retellings are not shortened copies of experience. The amount of temporal compression in verbal recall predicts systematic transformation between encoding and recall representations in high-level association cortex ([Musz & Chen, 2022](https://www.nature.com/articles/s42003-022-03418-5)). Narrative structure also influences what is accessible and remembered. Semantically and causally central events are recalled more often and produce stronger hippocampal and default-mode-network signatures ([Lee & Chen, 2022](https://www.nature.com/articles/s41467-022-31965-2)). These findings motivate decomposing summarization into structured transformations rather than measuring only word count or whole-text similarity.

Developmental work motivates target age as a continuous moderator. Event structure constrains information accessibility less in children than adults ([Ren et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S0010027721002997)), while memory reactivation and inferential integration differ across ages 7–30 ([Schlichting et al., 2022](https://www.nature.com/articles/s41562-021-01206-5)). Children’s retells can be decomposed into central and peripheral idea units, with age and narrative properties affecting what is recalled ([Russell, Wang, & Cain, 2024](https://www.tandfonline.com/doi/full/10.1080/10409289.2024.2303908)). The present study adapts these distinctions to model-generated summaries while avoiding the stronger claim that artificial target-age steering reproduces human development.

Content and temporal organization are treated separately. Event boundaries can improve content recognition while impairing temporal-order memory ([Zheng et al., 2022](https://www.nature.com/articles/s41593-022-01020-w)). Relevant distant events are selectively reactivated at narrative boundaries ([Hahamy, Dubossarsky, & Behrens, 2023](https://www.nature.com/articles/s41593-023-01331-6)), and cortico-hippocampal systems differently reuse representations of people, contexts, schemas, and specific events ([Reagh & Ranganath, 2023](https://www.nature.com/articles/s41467-023-36805-5)). Accordingly, a summary can preserve central content while changing chronology, or integrate distant details while remaining semantically grounded.

The mechanistic design follows recent work showing that representational and functional similarity are not interchangeable. A common framework for representational alignment explicitly separates representation, behavior, and intervention ([Sucholutsky et al., 2025](https://openreview.net/forum?id=Hiq7lUh4Yn)). Analytical results likewise demonstrate that networks can be functionally similar but representationally different, or representationally similar but functionally different ([Braun, Grant, & Saxe, 2025](https://proceedings.mlr.press/v267/braun25a.html)). Therefore behavioral, representational, and causal results are reported independently.

## Innovation

1. **Summary-complete alignment:** Every summary detail is evaluated. The analysis does not begin from source details and ignore summary details that fail to find a convenient match.
2. **Variable-resolution matching:** A summary detail can align to a contiguous source span of one to ten details. This measures whether one short summary unit integrates several source units.
3. **Explicit information inequality:** A capacitated transport model includes a null source instead of forcing every summary detail to match something. This follows work showing that null alignment is essential when texts contain unequal information ([Arase, Bao, & Yokoi, 2023](https://aclanthology.org/2023.acl-long.219/)).
4. **Centrality-conditioned survival:** The design compares retained and omitted source details while accounting for each detail’s semantic position in the narrative network.
5. **Behavior–representation dissociation:** Output-level developmental effects are tested separately from hidden-state geometry and layerwise age decoding.
6. **Mechanistic triangulation:** Residual patching, activation steering, router interventions, Jacobian-lens analysis, and sparse features test the same behavioral endpoint through distinct mechanisms.
7. **Cross-system measurement:** The detail-survival framework is designed to apply unchanged to model summaries and human recall transcripts.

## Data and experimental conditions

### Source stories

The source pool consists of cleaned YouTube story transcripts linked by `video_id` to extracted book metadata and Amazon reading-age estimates. The primary cohort includes stories with:

- a recoverable cleaned narrative;
- detail-level segmentation;
- a valid customer reading-age range;
- a stable story and video identifier; and
- at least one eligible age-conditioned summary.

Customer age is primary because it provides a consistent audience-age signal. Publisher age is retained for sensitivity analysis but is not silently mixed with customer age in the confirmatory cohort.

The final sample size is computed when `00_dataset_index` runs rather than being hard-coded in the proposal. This prevents stale counts as additional stories, ages, and segmented summaries are synchronized. All exclusions and structurally impossible conditions are recorded.

### Target-age conditions

For a source range `[L,U]`, the primary relative conditions are:

- **younger:** `L - 3`, when the resulting target is at least age 2;
- **actual:** the source reading-age range, truncated to ages 2–18 only when necessary;
- **older:** `U + 3`, if at most age 18; and
- **no age:** an optional native open-model control, not part of the confirmatory GPT cohort.

The relative design tests developmental movement around each book’s intended audience while keeping story content fixed. Boundary conditions are marked ineligible rather than replaced with a different contrast.

### Summary budget and model provenance

The confirmatory behavioral cohort retains every available valid younger, actual, and older summary. Its achieved compression ratio is measured rather than assumed and enters all age-effect models as a covariate and sensitivity stratum. The age-grid and variable-compression corpora are exploratory appendix analyses. Native open-model summaries match the observed GPT word budget within each story-condition so behavioral concordance is not confounded by unequal output length.

Two summary sources have different inferential roles:

1. **Closed-model summaries** generated by GPT-5.4-mini or GPT-5.6-luna provide the large natural-language behavioral corpus. Teacher-forcing these texts through an open model measures how that open model represents the completed summaries; it does not recover the generator’s hidden computation.
2. **Native open-model summaries** generated by the same model being intervened on provide the valid corpus for direct mechanistic causal claims.

The current complete younger/actual/older corpus has a broad compression distribution and is retained as the primary behavioral sample. Length and achieved compression ratio are always included as covariates or sensitivity filters; fixed-ratio analyses are robustness checks rather than replacements for the full cohort.

## Formal behavioral measures

Let source details be `x_1,...,x_n` and summary details be `y_1,...,y_m`.

### Semantic centrality

Each source detail is embedded and assigned off-diagonal weighted degree:

`C_i = (1 / (n - 1)) sum_{k != i} cos(e(x_i), e(x_k))`.

Centrality is normalized within story. It measures relational connectedness, not an externally validated label of literary importance. Analyses report both continuous centrality and preregistered percentile splits for central versus peripheral details.

### Structured summary alignment

Every summary detail is assigned a contiguous source span `w_j=[a_j,b_j]` of length 1–10. A monotonic dynamic program maximizes:

`sum_j [cos(e(y_j),e(w_j)) - lambda_len log2(|w_j|) - lambda_overlap O_j - lambda_reverse R_j]`.

The selected span length `|w_j|` is the **summarizing index**. A value of one indicates atomic retention; larger values indicate that one summary detail integrates several consecutive source details. The analysis also retains unpenalized best-span length, gain over the best single-detail match, and the best nonconsecutive source set of the same size.

### Atomic transport and null grounding

A second alignment solves a capacitated linear transport problem in which each summary detail sends one unit of mass to source details or to a null source. Source capacities prevent unlimited reuse. The null rate estimates summary content for which no source detail provides sufficient semantic support.

### Detail survival

A source detail is considered retained under several operationalizations:

- covered by a selected structured span;
- assigned at least 0.5 transport mass;
- above a prespecified cosine threshold; or
- retained under consensus across structured and transport definitions.

Threshold sensitivity is reported rather than selecting the threshold that maximizes the age effect.

### Narrative organization

Summary-level outcomes include:

- source-detail coverage;
- centrality-weighted coverage;
- retained-minus-omitted centrality;
- mean summarizing index;
- multi-detail summary rate;
- nonconsecutive fusion rate;
- null-grounding rate; and
- order-inversion rate based on aligned source positions.

Whole-summary/full-story cosine is reported as a coarse grounding measure, not as a substitute for detail-level alignment.

## Hypotheses

### Behavioral hypotheses

- **H1 — Selective compression:** At a within-story matched word budget, increasing target age will increase the summarizing index and favor integration of multiple source details rather than simple lexical deletion.
- **H2 — Centrality selection:** Central source details will survive more often than peripheral details. The centrality advantage will vary with target age if developmental steering changes which information is treated as essential.
- **H3 — Matched-age grounding:** Summaries targeted to the source audience will show stronger source grounding than equally distant mismatched conditions. This is tested independently from the monotonic older-versus-younger contrast.
- **H4 — Order preservation:** Age effects on semantic retention will not necessarily imply equivalent effects on chronology. Order-inversion and dispersed-fusion outcomes may reveal a separate developmental trajectory.

The direction of H2’s interaction and all nonlinear age effects are treated as empirical questions. Failure to find a monotonic slope does not imply no developmentally meaningful structure; matched-age peaks or local transitions may be present.

### Representational hypotheses

- **H5 — Layerwise trajectory:** Target age and age condition will be decodable from intermediate hidden states under leave-story-out validation.
- **H6 — Incremental representation:** Hidden states will improve prediction beyond behavioral features such as achieved compression, number of details, and source age.
- **H7 — Detail-survival code:** Original-detail activations will predict later survival in summaries from held-out stories.
- **H8 — Routing specialization:** For mixture-of-experts models, age condition and detail survival will alter router entropy, top-expert choice, or expert probability at selected layers.

### Causal hypotheses

- **H9 — Residual mediation:** Age-matched residual patches will selectively improve the likelihood of the corresponding summary, and age-direction steering will move generated information selection in the predicted direction.
- **H10 — Router contribution:** Router-only patching will explain a measurable but potentially partial component of the residual-stream effect.
- **H11 — Verbalizable intermediate state:** Jacobian-lens readouts will expose age-related intermediate content before the final layer, and coordinate swaps will causally alter summary behavior.
- **H12 — Sparse mediators:** A small set of reproducible SAE features will predict detail survival and causally alter retention when their decoder directions are intervened on.

## Experiments

### Experiment 00 — Cohort construction and provenance

**Question:** Are originals, summaries, ages, and model provenance linked without leakage or silent exclusions?

**Procedure:** Build one canonical record table containing originals and summaries, plus a condition-eligibility table. Deduplicate stable story-condition-summary combinations, preserve model provenance, compute actual compression, and record absent or ineligible conditions.

**Why:** Every later comparison depends on correct within-story pairing. A sophisticated mechanistic analysis cannot repair an incorrectly joined behavioral dataset.

**Primary outputs:** `cohort.jsonl`, `cohort_metadata.csv`, and `condition_index.csv`.

### Experiment 01 — Model and instrumentation smoke test

**Question:** Can the chosen open model expose hidden states, router logits, generation, and hookable layers under the HPC environment?

**Procedure:** Load one story, perform a forward pass and a short generation, verify hidden-state shapes, identify layer modules, and test whether router outputs are available.

**Why:** This is a technical gate, not a scientific result. It prevents large dependent jobs from running with incompatible model code.

### Experiment 13 — Structured detail alignment and behavioral phenotype

**Question:** What changes in the summaries as target age changes?

**Procedure:** Embed source details, summary details, contiguous source spans, and complete texts. Run monotonic dynamic-program alignment and null-capable transport. Compute survival, centrality, compression, grounding, and order metrics for every summary.

**Inference:**

- story fixed-effects models for continuous target-age effects;
- target age by absolute age-distance interactions;
- matched-age paired contrasts with story bootstrap confidence intervals;
- sign-flip randomization tests;
- conditional logistic regression within story-summary pair for centrality-conditioned survival; and
- sensitivity analysis across matching definitions and thresholds.

**Why first:** Mechanistic interpretation is only warranted if the output-level developmental effect is reproducible and robust to alignment choices.

### Experiment 02 — Layerwise detail-representation extraction

**Question:** Where are story details, target age, and summary condition represented?

**Procedure:** Teacher-force original and summary texts through GPT-OSS-20B or Qwen3-8B. Locate each detail in token space and mean-pool residual activations at layers `0,4,8,12,16,20,last`. Save per-detail vectors and router summaries when the architecture exposes them.

**Why:** Detail-level pooling preserves the unit used by the behavioral analysis. Whole-story pooling alone could hide selective information loss.

### Experiment 03 — Native budget-matched open-model generation

**Question:** Does the same open model show the behavioral trajectory when it generates its own summaries?

**Procedure:** Generate the exact younger, actual-age, and older story-condition pairs available in the closed-model cohort, matching each GPT summary's observed word budget. Fixed-ratio and no-age generations are separate controls.

**Why:** Closed-model outputs support behavioral analysis and receiver-model representation analysis. Native outputs are required for causal claims about the intervened model’s generation process.

### Experiment 04 — Layerwise probes and survival prediction

**Question:** Is developmental information linearly accessible, and does it generalize to unseen stories?

**Procedure:** At each layer, fit grouped classifiers for condition, regressors for target age, and classifiers for original-detail survival. Use `GroupKFold` by story, standardized features, shuffled labels, and behavioral covariates. Save older-minus-actual, actual-minus-younger, and older-minus-younger directions for interventions.

**Why:** Layerwise emergence constrains where causal interventions should be attempted. Grouped validation prevents details from the same story appearing in both train and test folds.

**Interpretation limit:** Probe success establishes accessibility, not use by the model.

### Experiment 10 — Behavioral–representational dissociation

**Question:** Do representations contain age information beyond visible summary behavior?

**Procedure:** Compare four held-out models: behavioral-only, representation-only, behavior-plus-representation, and shuffled-label control. Behavioral features include source age, achieved compression, number of details, and mean detail length.

**Why:** This is the direct test of the paper’s main dissociation. Possible outcomes include:

- behavior and representation both change;
- behavior changes with no additional representation signal;
- representation changes with little behavioral effect; or
- neither changes reliably.

### Experiment 07 — Expert-routing analysis

**Question:** Does developmental steering recruit different computational subcircuits in a mixture-of-experts model?

**Procedure:** Analyze router entropy, top-expert identity, top-expert probability, and expert-distribution divergence across layers and age conditions. Join these outcomes to detail survival in the final inferential table. Then patch router decisions between age conditions while leaving the residual stream otherwise intact and score the target summary by teacher-forced negative log likelihood.

**Why:** Routing offers a mechanistically distinct path from distributed residual geometry. Router-only effects can reveal conditional computation even when pooled hidden states appear similar.

### Experiment 08 — Competitive normalization and latent demixing

**Question:** Can detail selection be explained as competition among story-detail representations under a finite summary budget?

**Procedure:** Compare held-out survival models using raw activations, within-story divisively normalized activations, and low-dimensional PCA-plus-NMF components. The normalization model has the generic form:

`z_i = r_i / (sigma + sum_k w_ik r_k)`.

NMF components provide nonnegative latent mixtures that may separate centrality, position, concreteness, or age-sensitive content.

**Why:** This supplies a more explicit computational hypothesis than a spline over age. It tests whether information survival depends on a detail’s activation relative to competing details, not only its absolute activation.

The design is inspired by divisive normalization as a canonical neural computation ([Carandini & Heeger, 2012](https://www.nature.com/articles/nrn3136)) and by joint denoising/demixing approaches in neural data analysis ([Pnevmatikakis et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4881387/)). These are mathematical inspirations, not evidence that language models or narrative memory implement the same biological mechanism.

### Experiment 05 — Residual-stream causal patching

**Question:** Are age-associated residual states causally involved in summary selection?

**Procedure:** Patch actual-age residual states into younger or older prompts at candidate layers selected from Experiment 04. Score the observed younger or older summary by teacher-forced token-level negative log likelihood. Include unrelated-story states and norm-matched random states.

**Outcomes:** The implemented patching endpoint is change in sequence and token-level negative log likelihood. Direct changes in generated detail survival are tested by Experiments 06, 11, and 12, which generate intervened summaries and pass them through the behavioral alignment pipeline.

**Why:** If representation is causal, a matched intervention should move the behavioral endpoint more than unrelated-story or no-intervention controls.

### Experiment 06 — Activation steering and dose response

**Question:** Can an age direction continuously control compression behavior?

**Procedure:** Add the learned older-minus-actual or younger-minus-actual direction during generation at strengths `-2,-1,0,1,2`. Compare with a norm-matched random direction and repeat across candidate layers and prompt phrasings.

**Why:** A monotonic dose response is stronger evidence than a single patch. Reliability checks are necessary because steering vectors can be brittle and contaminated by dataset-specific features ([Tan et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html)).

### Experiment 09 — Directional next-token Jacobian

**Question:** Which output-token tendencies are locally sensitive to the age direction?

**Procedure:** Perturb selected residual layers by a small epsilon along an age direction and estimate the derivative of next-token logits by finite difference. Record the largest positive and negative token derivatives.

**Why:** This inexpensive analysis tests whether the direction has a coherent local effect before fitting a full Jacobian lens. It is diagnostic, not a substitute for sequence-level causal generation.

### Experiment 11 — Jacobian lens

**Question:** Does a verbalizable developmental workspace emerge across layers?

**Procedure:** Fit the corpus-averaged Jacobian lens from the reference implementation on Qwen3-8B, apply it at detail midpoints, and compare intermediate readouts across original, younger, actual, and older conditions. Intervene by swapping younger- and older-associated coordinates, with random matched swaps.

**Why:** Standard logit-lens readouts can mischaracterize intermediate residual states. The Jacobian lens estimates how intermediate directions map into downstream verbalizable content. It tests whether age-conditioned information is not only decodable but expressed in interpretable intermediate coordinates.

This experiment follows [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) and uses the associated [reference implementation](https://github.com/anthropics/jacobian-lens).

### Experiment 12 — Sparse-autoencoder feature discovery and intervention

**Question:** Are developmental compression and detail survival mediated by a sparse set of reusable features?

**Procedure:** Train domain-specific TopK SAEs on token-level residual activations from the exact post-trained Qwen3-8B used for inference. Use official 65,536-feature TopK Qwen-Scope SAEs from `Qwen3-8B-Base` as a transfer sensitivity analysis, conditional on reconstruction quality. Report reconstruction cosine, mean L0, dead-feature fraction, condition effects, top activating examples, and leave-story-out survival prediction. Intervene on decoder directions derived from younger- and older-associated features, with random matched controls.

**Why:** Sparse features may separate superposed content that pooled residual probes conflate. However, feature labels are hypotheses; causal intervention and held-out prediction are required before describing a feature as a developmental compression mechanism. The design follows recent SAE scaling and evaluation work ([Gao et al., 2024](https://arxiv.org/abs/2406.04093); [Lieberum et al., 2024](https://arxiv.org/abs/2408.05147); [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)).

## Statistical analysis plan

### Units and dependence

The study has three nested units:

- source details within story;
- summary details within story-summary generation; and
- summary generations within story and condition.

Stories, not details, define independent cross-validation groups and bootstrap units. Standard errors are clustered by story. This prevents thousands of details from being treated as thousands of independent narratives.

### Confirmatory behavioral models

For source-detail survival, the core model is:

`logit P(S_isk = 1) = alpha_sk + beta_1 C_is + beta_2 C_is * A_sk + beta_3 P_is`,

where `alpha_sk` is a story-summary-pair stratum, `C_is` is within-story centrality, `A_sk` is target age, and `P_is` is normalized source position. The interaction tests whether target age changes centrality-based selection.

For summary-level metric `M_sk`:

`M_sk = alpha_s + beta_1 A_sk + beta_2 |A_sk - A_s| + beta_3 A_sk|A_sk-A_s| + epsilon_sk`.

Matched-age contrasts compare actual-age summaries with eligible younger and older summaries from the same story. Story bootstrap confidence intervals and sign-flip randomization p-values accompany model-based estimates.

### Nonlinearity

The relative younger/actual/older design is confirmatory. A secondary analysis fits restricted cubic splines over target age with story effects and visualizes target age against absolute age distance. Knots and degrees of freedom are fixed before examining outcomes. This tests gradual versus locally accelerated developmental change without searching over arbitrary age bins.

### Representation models

Layerwise target-age regression and condition classification use grouped cross-validation. Performance is compared across:

- behavioral features;
- representation features;
- combined features; and
- shuffled labels.

Incremental value is the held-out difference between combined and behavioral-only performance, not raw training-set decodability.

### Causal contrasts

For generation interventions, the primary effect is:

`Delta = [Y(age_direction, alpha) - Y(clean)] - [Y(random_matched, alpha) - Y(clean)]`.

Dose-response slopes, direction reversibility, layer specificity, and prompt robustness are reported. An intervention is considered supportive only if it changes the preregistered behavioral measure, exceeds the matched control, and generalizes across held-out stories.

For residual and router patches that use teacher-forced scoring, the endpoint is instead a difference in negative log likelihood relative to unrelated-story, random-state, and clean conditions. Such a likelihood effect is evidence about sequence preference, not by itself evidence that free generation would preserve different details.

### Multiplicity and robustness

- Primary outcomes are source-detail survival, summarizing index, centrality-weighted coverage, null rate, and order-inversion rate.
- Benjamini–Hochberg false-discovery-rate correction is applied within each experiment family across layers and outcomes.
- Exact duplicate summaries are removed; distinct regenerated variants remain separate but clustered within story.
- Results are repeated under alternative retention definitions, embedding models, length penalties, null thresholds, source capacities, and maximum span sizes.
- Customer-age results are primary; publisher-age and combined-source analyses are labeled sensitivity analyses.

### Implementation boundary

Experiments 00–17, their HPC wrappers, the linear within-story models, bootstrap/sign-flip contrasts, grouped probes, fold-local demixing, psycholinguistic features, Möbius integration, train/evaluation isolation, family-level FDR adjustment, and primary causal controls are implemented. Restricted cubic splines, the router-to-survival inferential join, and the human-recall bridge are prespecified proposal analyses that still require their final reporting scripts. This distinction prevents planned analyses from being mistaken for completed results.

## Decision logic and falsification criteria

The project is intentionally contingent:

1. **No robust behavioral age effect:** Stop claims about a developmental compression trajectory. Report alignment and centrality results as descriptive measurement work only.
2. **Behavioral effect without representation signal:** Conclude that output adaptation is not captured by the tested receiver-model geometry, rather than claiming no representation exists.
3. **Representation signal without incremental prediction:** Treat age as decodable but behaviorally redundant.
4. **Representation and behavior without causal intervention effect:** Describe correlation, not mediation.
5. **Age intervention equals random control:** Reject specificity of the proposed direction or feature set.
6. **Causal effects only on closed-model teacher-forced text:** Do not attribute those effects to the closed generator. Require native open-model replication.
7. **SAE features predict but do not intervene:** Treat features as markers, not mechanisms.

## Human behavioral bridge

The planned human dataset contains participant recall text and participant age, initially over a small number of stories and approximately ages 8–14. Each source story and recall will be segmented into the same content-unit definition used for model summaries. The alignment pipeline will produce:

- recall survival for each source detail;
- retained-versus-omitted centrality;
- number of source details integrated per recall unit;
- unsupported additions;
- chronology and inversion measures; and
- participant-level compression profiles.

The primary translational test is whether model-derived source-detail predictors improve held-out human recall prediction beyond centrality, position, story identity, and participant age. With only one or a few stories, claims will be limited to proof of concept; uncertainty will be estimated at the participant level, and story-general developmental claims will wait for additional narratives.

## Expected outcomes

The strongest supporting pattern would be:

1. stable within-story age effects on detail survival and summarizing index;
2. stronger central-detail selection at some target ages, with chronology changing independently;
3. target-age information emerging at intermediate layers and improving held-out prediction beyond behavioral measures;
4. corresponding changes in router use, J-lens readouts, or sparse features; and
5. selective causal movement of the same behavioral outcomes under age-direction interventions.

Scientifically informative dissociations are equally important. For example, summaries may differ in centrality selection while hidden-state age decoding remains weak, or a strong internal age axis may fail to control which details survive. Such outcomes directly test the broader claim that aligned model behavior need not imply aligned representation.

## Risks and alternatives

- **Age labels are noisy:** Separate customer and publisher sources, retain ranges rather than only midpoints, and test source-specific robustness.
- **Summary length confounds age:** Match native outputs to each available GPT story-condition budget, record achieved compression, and include achieved length in sensitivity models.
- **Embedding similarity forces false matches:** Use null transport, threshold sensitivity, multiple embedding models, and human review of stratified samples.
- **Long spans win trivially:** Penalize span length and report raw versus penalized optima and gain over single-detail matches.
- **Probe leakage across related details:** Split by story, never by detail row.
- **Cross-model attribution error:** Reserve causal claims for native generations of the intervened model.
- **Steering is unstable:** Require dose response, direction reversal, prompt robustness, and random controls.
- **SAE features are uninterpretable or dead:** Report reconstruction and sparsity diagnostics, use official and domain-trained SAEs, and retain distributed residual analyses as the fallback.
- **Router logits are unavailable:** Treat the smoke test as a gate and omit router claims rather than inferring routing from hidden states.
- **Human data are too sparse:** Use the model work to define measurement hypotheses, but report human results as exploratory until multiple stories and participants per age are available.

## Reproducibility and computational plan

All runs use one canonical cohort index and write large outputs under `mechanistic_interpretability/outputs/`. HPC jobs are dependency-linked so that extraction precedes probes and probes precede causal interventions. Large activation arrays remain on scratch; aggregate reports are synchronized to Git.

The principal GPU workloads are:

- 15 representation-extraction shards, up to 12 hours each;
- 13 native-generation shards, up to 12 hours each;
- residual, steering, router, and Jacobian interventions on one GPU each;
- 10 Jacobian-lens fitting shards and 20 application shards on Qwen3-8B; and
- SAE collection, official-checkpoint analysis, domain training, and intervention at layers 8, 16, and 24.

CPU analyses include structured alignment statistics, grouped probes, routing summaries, normalization/demixing, and behavior–representation comparisons. Seeds, model identifiers, layers, age conditions, thresholds, compression ratios, and exclusion reasons are saved with outputs.

## Work plan

### Phase 1 — Measurement validation

1. Rebuild the canonical cohort and verify counts and age-condition coverage.
2. Run the structured-alignment validation checks.
3. Hand-review central, peripheral, null, one-to-many, and order-inversion examples.
4. Freeze primary thresholds and outcomes.

### Phase 2 — Behavioral developmental analysis

1. Run Experiment 13 over all eligible summaries.
2. Estimate within-story age effects, matched-age contrasts, and centrality selection.
3. Run nonlinear and threshold sensitivity analyses.
4. Establish whether there is a behavioral phenomenon worth mechanistically explaining.

### Phase 3 — Representation mapping

1. Run the model smoke test.
2. Extract layerwise detail vectors and router summaries.
3. Run grouped probes and behavior–representation dissociation.
4. Use preregistered layers for confirmatory intervention; any data-driven layer selection occurs only within the training partition or a nested validation split, never on the final evaluation stories.

### Phase 4 — Causal mechanism tests

1. Run residual patching and activation steering.
2. Run router-only patching.
3. Run the directional Jacobian diagnostic.
4. Fit and apply the Jacobian lens, then run coordinate interventions.
5. Analyze official and domain SAEs, predict survival, and intervene on age-associated features.

### Phase 5 — Cross-system validation and paper synthesis

1. Apply the same alignment outcomes to human recalls.
2. Compare model and human centrality-survival functions without assuming representational identity.
3. Report convergent and dissociative findings under the contingent claim structure.

## Anticipated contribution

The project will contribute a rigorous way to study developmental information compression in natural-language systems. Its main value is not a claim that language models develop like children. It is a testable framework for discovering how age instructions reorganize information selection, whether observable adaptation corresponds to an internal trajectory, and which internal components causally control the survival of narrative details. The same behavioral measures can then be applied to scarce human recall data, allowing mechanistic model analysis to generate precise hypotheses for developmental cognition while keeping artificial and biological representations conceptually distinct.
