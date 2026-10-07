# Story Perplexity Age-Order Analysis

This folder scores each story under long-context open-weight causal LMs. Quantized variants are intentionally omitted because the related `unsteerable-transformer` quantized runs did not work reliably.

The dataset label is `age_median`: the midpoint of the Amazon reading-age range. For example, `4-8 years` becomes `6.0`. Open-ended ranges such as `12+` use `18` as the default upper bound.

The default scoring mode is whole-story: `MAX_LENGTH=0` uses the registry context window, and `OVERLENGTH_POLICY=skip` flags any story that does not fit instead of truncating it. Output columns `fits_single_window` and `n_windows` make this explicit.

Use a recent Transformers install on HPC. Qwen3 long-context loading needs
`transformers>=4.51.0`; Gemma checkpoints also require that your Hugging Face
account has accepted the Gemma terms.

## 1. Prepare Stories

```bash
python src/analysis/natural_language/perplexity/prepare_dataset.py \
  --master data/processed/master_stories.parquet \
  --out results/analysis/natural_language/perplexity/stories_for_perplexity.parquet \
  --dedupe
```

If the master table needs rebuilding first:

```bash
python src/data_pipeline/cleaning/master_dataset/build_master_dataset.py
```

By default this uses `story_wAmazonData_042626/amazon_youtube_batch1_wid.csv` as the
story spine and `story_wAmazonData_042626/cleaned_idchecked/*.txt` as the
cleaned text source. It writes `data/processed/master_stories.parquet`,
`data/processed/master_stories.csv.gz`, and `data/processed/master_stories_index.csv`.

If you want to ship the cleaned text as one archive instead of 1,733 loose
files, create a zip and point the builder or Slurm job at it:

```bash
python -m zipfile -c data/processed/cleaned_idchecked.zip story_wAmazonData_042626/cleaned_idchecked

python src/data_pipeline/cleaning/master_dataset/build_master_dataset.py \
  --transcripts-zip data/processed/cleaned_idchecked.zip \
  --transcripts-source-name cleaned_idchecked
```

## 2. Run One Model On HPC

```bash
MODEL_KEY=qwen3_8b sbatch hpc/analysis/natural_language/perplexity/run_model_perplexity.slurmnatural_language/perplexity/run_model_perplexity.slurm
```

If using a zip archive on HPC:

```bash
TRANSCRIPTS_ZIP=data/processed/cleaned_idchecked.zip \
TRANSCRIPTS_SOURCE_NAME=cleaned_idchecked \
MODEL_KEY=qwen3_8b \
sbatch hpc/analysis/natural_language/perplexity/run_model_perplexity.slurmnatural_language/perplexity/run_model_perplexity.slurm
```

The Slurm wrapper rebuilds the master file, prepares the src/analysis/natural_language/perplexity dataset, and
then scores it. Defaults:

- `BASE=story_wAmazonData_042626/amazon_youtube_batch1_wid.csv`
- `TRANSCRIPTS_DIR=story_wAmazonData_042626/cleaned_idchecked`
- `MASTER_OUT=data/processed/master_stories.parquet`
- `PPL_DATASET=results/analysis/natural_language/perplexity/stories_for_perplexity.parquet`
- `OVERLENGTH_POLICY=skip`

Useful overrides:

```bash
MODEL_KEY=olmo_2_7b_base LIMIT=100 sbatch hpc/analysis/natural_language/perplexity/run_model_perplexity.slurmnatural_language/perplexity/run_model_perplexity.slurm
```

## 3. Submit All Registered Models

```bash
bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.shnatural_language/perplexity/submit_all_perplexity.sh
```

By default, this submits every registered model using Torch account
`torch_pr_494_general`, the CUDA 12.3 Singularity image, and the same
`story-features` overlay used by the older scripts:
`/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro`.
Each model job requests one generic GPU, 4 CPU cores, 24G CPU memory, and 4
hours walltime.
Override these only if your HPC setup differs:

```bash
ACCOUNT=torch_pr_494_general \
OVERLAY=/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
CONDA_ENV=story-features \
PYTHON_BIN=/ext3/story-features/bin/python \
bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.shnatural_language/perplexity/submit_all_perplexity.sh
```

To override walltime or CPU memory at submit time:

```bash
TIME=02:00:00 MEM=24G bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.shnatural_language/perplexity/submit_all_perplexity.sh
```

Use `MODEL_KEYS` only if you need to manually restrict a rerun:

```bash
MODEL_KEYS="qwen3_8b olmo_2_7b_base" bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.shnatural_language/perplexity/submit_all_perplexity.sh
```

To inspect the registry:

```bash
python -m src/analysis/natural_language/perplexity.models --tsv
```

To inspect the analysis comparison sets:

```bash
python -m src/analysis/natural_language/perplexity.models --list-comparison-sets
```

The current registry emphasizes:

- Qwen3 8B, 14B, 30B-A3B, and 32B: native 32K/256K depending on checkpoint; the 32K checkpoints are loaded with YaRN for 128K scoring.
- Llama 3.1 8B and Llama 3.3 70B: text-only Llama anchors with 128K context. These require accepted Meta gated-model access.
- NVIDIA Llama-Nemotron Nano 8B and Super 49B: 128K-context reasoning models. The 49B checkpoint uses custom code and should be treated as a large-HPC run.
- Kimi K2 Instruct 0905: 256K-context MoE, 1T total / 32B active. This is a very large custom-code run.
- Mistral-Nemo 12B: runnable text-only Mistral family anchor. Current Mistral Small 3.2 is newer, but uses Mistral3/vLLM-style loading and is not included in this causal-LM scorer.
- OLMo 2 1B, 7B, 13B, and 32B base/instruct: included to mirror the `Learning is Forgetting` model family, but these are 4K-context models, so overlength stories are skipped rather than truncated.
- Gemma 3 1B base/instruct: current text-only Gemma 3 checkpoints with 32K context. Larger Gemma 3 checkpoints are multimodal and need a separate scorer.

Comparison sets are for analysis after the model outputs exist:

```bash
python src/analysis/natural_language/perplexity/analyze_age_order.py \
  --runs results/analysis/natural_language/perplexity/runs \
  --outdir results/analysis/natural_language/perplexity/analysis \
  --metric mean_nll \
  --comparison-set similar_scale
```

To run only long-context models and exclude OLMo 2:

```bash
MIN_CONTEXT_LENGTH=32768 bash hpc/analysis/natural_language/perplexity/submit_all_perplexity.shnatural_language/perplexity/submit_all_perplexity.sh
```

Quantized checkpoints are not part of this runner. The earlier bitsandbytes
path failed in the related unsteerable runs, and GGUF/FP8 server deployments
need a separate server-side src/analysis/natural_language/perplexity scorer to keep scores comparable.

Rough walltime estimates for the full 2,075-story dataset, assuming the model is
already cached on `/scratch/$USER/hf`:

- 1B models: about 10-30 minutes.
- 7B-14B models: about 30-90 minutes.
- 30B-32B models: about 1.5-4 hours.
- 49B, 70B, and Kimi K2: not realistic under the default one generic GPU / 24G CPU memory job; run these separately only on appropriate large-GPU or multi-GPU resources.

## 4. Analyze Age Order

```bash
python src/analysis/natural_language/perplexity/analyze_age_order.py \
  --runs results/analysis/natural_language/perplexity/runs \
  --outdir results/analysis/natural_language/perplexity/analysis \
  --metric mean_nll
```

Use `mean_nll` or `log10_perplexity` for correlations. Raw `src/analysis/natural_language/perplexity` is usually heavily skewed.

Main outputs:

- `results/analysis/natural_language/perplexity/runs/<model_key>.csv`: one row per story per model.
- `results/analysis/natural_language/perplexity/analysis/age_order_summary_mean_nll.csv`: model-level Spearman/Pearson age-order summary.
- `results/analysis/natural_language/perplexity/analysis/age_bins_mean_nll.csv`: per-age median/mean src/analysis/natural_language/perplexity table.
- `results/analysis/natural_language/perplexity/analysis/age_order_summary_<comparison_set>_mean_nll.csv`: same summary restricted to a named comparison set.

## 5. Plot Age Geometry

To make a paper-style sequential age-geometry plot, first run the analysis above,
then:

```bash
python src/analysis/natural_language/perplexity/plot_age_geometry.py \
  --age-bins results/analysis/natural_language/perplexity/analysis/age_bins_mean_nll.csv \
  --outdir results/analysis/natural_language/perplexity/figures
```

On the HPC login node without system Python, run it inside the same Singularity
overlay:

```bash
export SIF=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
export OVERLAY=/scratch/$USER/my_env/overlay-15GB-500K.ext3:ro
export CONDA_ENV=story-features
export PYTHON_BIN=/ext3/story-features/bin/python
export USE_NV=0

./hpc/run_in_singularity.sh \
  'cd "$PWD" && "$PYTHON_BIN" src/analysis/natural_language/perplexity/plot_age_geometry.py \
    --age-bins results/analysis/natural_language/perplexity/analysis/age_bins_mean_nll.csv \
    --outdir results/analysis/natural_language/perplexity/figures'
```

This writes:

- `results/analysis/natural_language/perplexity/figures/age_geometry_mds_metric_median.png`: MDS age manifold plus scaled-isometry and neighbor-similarity panels.
- `results/analysis/natural_language/perplexity/figures/age_geometry_model_curves_metric_median.png`: per-model age curves.
- `results/analysis/natural_language/perplexity/figures/age_geometry_summary_metric_median.csv`: distance-correlation summary.
- `results/analysis/natural_language/perplexity/figures/age_geometry_pairwise_metric_median.csv`: pairwise age distances.

## 6. GitHub / HPC Handoff

The current push-ready data path is:

- `story_wAmazonData_042626/amazon_youtube_batch1_wid.csv`: 2,138-story spine.
- `story_wAmazonData_042626/cleaned_idchecked/`: 1,733 cleaned text files.
- `data/processed/master_stories.parquet`: rebuilt master table.
- `data/processed/master_stories.csv.gz`: CSV copy of the master table.
- `data/processed/master_stories_index.csv`: lightweight index.
- `results/analysis/natural_language/perplexity/stories_for_perplexity.parquet`: 2,075 labeled stories after the default word-count filter.

Recommended add list for this work:

```bash
git add \
  .gitignore \
  src/data_pipeline/cleaning/master_dataset/build_master_dataset.py \
  src/features/summaries/generate_prompt_condition_summaries.py \
  src/features/summaries/generate_open_weight_summaries.py \
  src/features/summaries \
  src/analysis/natural_language/perplexity \
  data/processed/master_stories.parquet \
  data/processed/master_stories.csv.gz \
  data/processed/master_stories_index.csv \
  results/analysis/natural_language/perplexity/stories_for_perplexity.parquet \
  story_wAmazonData_042626/amazon_youtube_batch1_wid.csv \
  story_wAmazonData_042626/cleaned_idchecked
```

Then commit and push:

```bash
git commit -m "Add long-context story src/analysis/natural_language/perplexity workflow"
git push
```
