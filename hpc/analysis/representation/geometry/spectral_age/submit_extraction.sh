#!/usr/bin/env bash
# Build the batch-video cohort and submit the two extraction stages for one model (default qwen3_8b).
#   MODEL_KEY=qwen3_8b bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-torch_pr_494_general}"
export MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
export COHORT="$ROOT/spectral_age_cohort.jsonl"
PERSHARD="${RECORDS_PER_SHARD:-100}"
export RECORDS_PER_SHARD="$PERSHARD"

# stdlib-only script: runs directly on the login node (no container, no SLURM_SUBMIT_DIR needed)
PYTHONPATH=src python3 -m analysis.representation.geometry.spectral_age.filter_cohort --output "$COHORT"
RECORDS=$(wc -l < "$COHORT")
SHARDS=$(( (RECORDS + PERSHARD - 1) / PERSHARD ))
echo "$RECORDS records -> $SHARDS shards of $PERSHARD"
GLOBAL_JOB=$(STAGE=global sbatch --parsable --account="$ACCOUNT" --array=0-$((SHARDS - 1))%4 --export=ALL hpc/analysis/representation/geometry/spectral_age/extract_subset.slurm)
DETAIL_JOB=$(STAGE=detail sbatch --parsable --account="$ACCOUNT" --array=0-$((SHARDS - 1))%4 --dependency=aftercorr:"$GLOBAL_JOB" --export=ALL hpc/analysis/representation/geometry/spectral_age/extract_subset.slurm)
echo "global job $GLOBAL_JOB, detail job $DETAIL_JOB (starts per shard after the matching global shard)"
