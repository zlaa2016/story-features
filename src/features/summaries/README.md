# Running Story Summary Jobs

This folder contains helpers for running summary-generation jobs without keeping an
interactive terminal open.

Use `gpt-4o-mini` as the main summary generator if that is the condition you want
to study. Use the open-weight long-context jobs as a generator-control /
sensitivity check, not as a silent replacement for the GPT-4o-mini summary set.

## tmux

Use this when running on a local machine or server where `tmux` is available.

Dry run:

```bash
bash hpc/features/summaries/run_tmux_summary_job.sh --limit 1 --dry-run
```

Real run:

```bash
export OPENAI_API_KEY="..."
bash hpc/features/summaries/run_tmux_summary_job.sh --limit 50 --model gpt-4o-mini
```

Reconnect:

```bash
tmux attach -t story_summaries
```

Use a custom session name:

```bash
SESSION_NAME=story_summaries_batch2 bash hpc/features/summaries/run_tmux_summary_job.sh --start 50 --limit 50
```

## Slurm: GPT-4o-mini API Baseline

Use this on an HPC cluster with `sbatch`. This is CPU-only because it calls the
OpenAI API.

Dry run:

```bash
mkdir -p logs/summary_prompt_runs
SUMMARY_ARGS="--limit 1 --model gpt-4o-mini --dry-run" \
  sbatch hpc/features/summaries/run_story_summaries.sbatchsummaries/run_story_summaries.sbatch
```

Real run:

```bash
export OPENAI_API_KEY="..."
SUMMARY_ARGS="--start 0 --limit 50 --model gpt-4o-mini" \
  sbatch hpc/features/summaries/run_story_summaries.sbatchsummaries/run_story_summaries.sbatch
```

Continue with the next batch:

```bash
export OPENAI_API_KEY="..."
SUMMARY_ARGS="--start 50 --limit 50 --model gpt-4o-mini" \
  sbatch hpc/features/summaries/run_story_summaries.sbatchsummaries/run_story_summaries.sbatch
```

## Slurm: Long-Context Open-Weight Summaries

These jobs run on one GPU through the existing Singularity wrapper. They use the
same prompt templates and output JSONL schema as the OpenAI API job.

Recommended GPT-style open-weight check:

```bash
OPEN_WEIGHT_SUMMARY_MODEL_KEY=gpt_oss_20b \
SUMMARY_START=0 \
SUMMARY_LIMIT=50 \
sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

H100/80GB-class GPT-style check:

```bash
OPEN_WEIGHT_SUMMARY_MODEL_KEY=gpt_oss_120b \
SUMMARY_START=0 \
SUMMARY_LIMIT=50 \
sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

Current long-context open-weight controls:

```bash
OPEN_WEIGHT_SUMMARY_MODEL_KEY=qwen3_8b \
SUMMARY_START=0 \
SUMMARY_LIMIT=50 \
sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

```bash
OPEN_WEIGHT_SUMMARY_MODEL_KEY=qwen3_30b_a3b_instruct_2507 \
SUMMARY_START=0 \
SUMMARY_LIMIT=50 \
sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

OLMo 2 paper-family control. These models have 4K context, so long stories may
be skipped by the overlength policy:

```bash
OPEN_WEIGHT_SUMMARY_MODEL_KEY=olmo_2_7b_instruct \
SUMMARY_START=0 \
SUMMARY_LIMIT=50 \
sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

Dry run without loading the model:

```bash
SUMMARY_ARGS="--model-key gpt_oss_20b --start 0 --limit 1 --dry-run" \
  sbatch hpc/features/summaries/run_open_weight_summaries.slurm
```

Submit the default open-weight comparison set:

```bash
bash hpc/features/summaries/submit_open_weight_summary_models.sh
```

Restrict the comparison set:

```bash
MODEL_KEYS="gpt_oss_20b qwen3_8b olmo_2_7b_instruct" \
  bash hpc/features/summaries/submit_open_weight_summary_models.sh
```

List available local model keys:

```bash
python src/features/summaries/generate_open_weight_summaries.py --list-models
```

Check status:

```bash
squeue -u "$USER"
```

Cancel:

```bash
scancel <JOBID>
```

Open-weight outputs are written under:

```bash
results/features/summaries/runs/summary_open_weight_<model_key>_<model_name>_<timestamp>/summary_outputs.jsonl
```

OpenAI API outputs are written under:

```bash
results/features/summaries/runs/summary_prompt_test_<model>_<timestamp>/summary_outputs.jsonl
```
