# Story Perplexity Age-Order Analysis

This folder scores each story under long-context open-weight causal LMs. Quantized variants are intentionally omitted because the related `unsteerable-transformer` quantized runs did not work reliably.

The dataset label is `age_median`: the midpoint of the Amazon reading-age range. For example, `4-8 years` becomes `6.0`. Open-ended ranges such as `12+` use `18` as the default upper bound.

The default scoring mode is whole-story: `MAX_LENGTH=0` uses the registry context window, and `OVERLENGTH_POLICY=skip` flags any story that does not fit instead of truncating it. Output columns `fits_single_window` and `n_windows` make this explicit.

Use a recent Transformers install on HPC. Qwen3 long-context loading needs
`transformers>=4.51.0`; Gemma checkpoints also require that your Hugging Face
account has accepted the Gemma terms.

## 1. Prepare Stories

```bash
python perplexity/prepare_dataset.py \
  --master data/processed/master_stories.parquet \
  --out artifacts/perplexity/stories_for_perplexity.parquet \
  --dedupe
```

If the master table needs rebuilding first:

```bash
python scripts/build_master_dataset.py
```

By default this uses `story_wAmazonData_042626/amazon_youtube_batch1_wid.csv` as the
story spine and `story_wAmazonData_042626/cleaned_idchecked/*.txt` as the
cleaned text source. It writes `data/processed/master_stories.parquet`,
`data/processed/master_stories.csv.gz`, and `data/processed/master_stories_index.csv`.

If you want to ship the cleaned text as one archive instead of 1,733 loose
files, create a zip and point the builder or Slurm job at it:

```bash
python -m zipfile -c data/processed/cleaned_idchecked.zip story_wAmazonData_042626/cleaned_idchecked

python scripts/build_master_dataset.py \
  --transcripts-zip data/processed/cleaned_idchecked.zip \
  --transcripts-source-name cleaned_idchecked
```

## 2. Run One Model On HPC

```bash
MODEL_KEY=qwen3_8b sbatch perplexity/run_model_perplexity.slurm
```

If using a zip archive on HPC:

```bash
TRANSCRIPTS_ZIP=data/processed/cleaned_idchecked.zip \
TRANSCRIPTS_SOURCE_NAME=cleaned_idchecked \
MODEL_KEY=qwen3_8b \
sbatch perplexity/run_model_perplexity.slurm
```

The Slurm wrapper rebuilds the master file, prepares the perplexity dataset, and
then scores it. Defaults:

- `BASE=story_wAmazonData_042626/amazon_youtube_batch1_wid.csv`
- `TRANSCRIPTS_DIR=story_wAmazonData_042626/cleaned_idchecked`
- `MASTER_OUT=data/processed/master_stories.parquet`
- `PPL_DATASET=artifacts/perplexity/stories_for_perplexity.parquet`
- `OVERLENGTH_POLICY=skip`

Useful overrides:

```bash
MODEL_KEY=olmo_2_7b_base LIMIT=100 sbatch perplexity/run_model_perplexity.slurm
```

## 3. Submit All Registered Models

```bash
bash perplexity/submit_all_perplexity.sh
```

To inspect the registry:

```bash
python -m perplexity.models --tsv
```

The current default registry emphasizes:

- Qwen3 8B, 14B, 30B-A3B, and 32B: native 32K/256K depending on checkpoint; the 32K checkpoints are loaded with YaRN for 128K scoring.
- OLMo 2 1B, 7B, 13B, and 32B base/instruct: included to mirror the `Learning is Forgetting` model family, but these are 4K-context models, so overlength stories are skipped rather than truncated.
- Gemma 3 1B base/instruct: current text-only Gemma 3 checkpoints with 32K context. Larger Gemma 3 checkpoints are multimodal and need a separate scorer.

To run only long-context models and exclude OLMo 2:

```bash
MIN_CONTEXT_LENGTH=32768 bash perplexity/submit_all_perplexity.sh
```

## 4. Analyze Age Order

```bash
python perplexity/analyze_age_order.py \
  --runs artifacts/perplexity/runs \
  --outdir artifacts/perplexity/analysis \
  --metric mean_nll
```

Use `mean_nll` or `log10_perplexity` for correlations. Raw `perplexity` is usually heavily skewed.

Main outputs:

- `artifacts/perplexity/runs/<model_key>.csv`: one row per story per model.
- `artifacts/perplexity/analysis/age_order_summary_mean_nll.csv`: model-level Spearman/Pearson age-order summary.
- `artifacts/perplexity/analysis/age_bins_mean_nll.csv`: per-age median/mean perplexity table.

## 5. GitHub / HPC Handoff

The current push-ready data path is:

- `story_wAmazonData_042626/amazon_youtube_batch1_wid.csv`: 2,138-story spine.
- `story_wAmazonData_042626/cleaned_idchecked/`: 1,733 cleaned text files.
- `data/processed/master_stories.parquet`: rebuilt master table.
- `data/processed/master_stories.csv.gz`: CSV copy of the master table.
- `data/processed/master_stories_index.csv`: lightweight index.
- `artifacts/perplexity/stories_for_perplexity.parquet`: 2,075 labeled stories after the default word-count filter.

Recommended add list for this work:

```bash
git add \
  .gitignore \
  scripts/build_master_dataset.py \
  scripts/generate_prompt_condition_summaries.py \
  scripts/generate_open_weight_summaries.py \
  scripts/summary_runs \
  perplexity \
  data/processed/master_stories.parquet \
  data/processed/master_stories.csv.gz \
  data/processed/master_stories_index.csv \
  artifacts/perplexity/stories_for_perplexity.parquet \
  story_wAmazonData_042626/amazon_youtube_batch1_wid.csv \
  story_wAmazonData_042626/cleaned_idchecked
```

Then commit and push:

```bash
git commit -m "Add long-context story perplexity workflow"
git push
```
