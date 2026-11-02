#!/usr/bin/env bash
# Build the batch-video cohort and submit the two extraction stages for one model (default qwen3_8b).
#   MODEL_KEY=qwen3_8b bash hpc/analysis/representation/geometry/spectral_age/submit_extraction.sh
# Rerun only some failed shards of an earlier submission: repeat that submission's variables (MODEL_KEY, COHORT_NAME, START_OFFSET, SHARD_OFFSET)
# and add ONLY_TASKS="20,21" (the array task numbers).  The cohort file is reused as it is, never rebuilt, so the shards keep their records.
set -euo pipefail
ACCOUNT="${ACCOUNT:-torch_pr_494_general}"
export MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
ROOT="results/features/representation_extraction/causal_lm/detail_extraction"
export COHORT="$ROOT/${COHORT_NAME:-spectral_age_cohort.jsonl}"
PERSHARD="${RECORDS_PER_SHARD:-100}"
export RECORDS_PER_SHARD="$PERSHARD"

# stdlib-only script: runs directly on the login node (no container, no SLURM_SUBMIT_DIR needed)
# VIDEOS_CSV / SUPPLEMENT: another video list and its supplement cohort (default: the frozen extraction list)
# EXCLUDE_DIRS: folders whose videos already have vectors (GPT-OSS: the existing global folder); START_OFFSET/SHARD_OFFSET: continue an earlier submission (records before START_OFFSET were already submitted; new shard ids start at SHARD_OFFSET)
if [[ -n "${ONLY_TASKS:-}" ]]; then
  [[ -f "$COHORT" ]] || { echo "ONLY_TASKS needs the existing cohort $COHORT" >&2; exit 1; }
else
  PYTHONPATH=src python3 -m analysis.representation.geometry.spectral_age.filter_cohort --output "$COHORT" ${VIDEOS_CSV:+--videos $VIDEOS_CSV} ${SUPPLEMENT:+--supplement $SUPPLEMENT} ${EXCLUDE_DIRS:+--exclude-dirs $EXCLUDE_DIRS}
fi
export START_OFFSET="${START_OFFSET:-0}" SHARD_OFFSET="${SHARD_OFFSET:-0}"
RECORDS=$(( $(wc -l < "$COHORT") - START_OFFSET ))
SHARDS=$(( (RECORDS + PERSHARD - 1) / PERSHARD ))
echo "$RECORDS records -> $SHARDS shards of $PERSHARD"
ARRAY="${ONLY_TASKS:-0-$((SHARDS - 1))}%4"
GLOBAL_JOB=$(STAGE=global sbatch --parsable --account="$ACCOUNT" --array="$ARRAY" --export=ALL hpc/analysis/representation/geometry/spectral_age/extract_subset.slurm)
DETAIL_JOB=$(STAGE=detail sbatch --parsable --account="$ACCOUNT" --array="$ARRAY" --dependency=aftercorr:"$GLOBAL_JOB" --export=ALL hpc/analysis/representation/geometry/spectral_age/extract_subset.slurm)
echo "global job $GLOBAL_JOB, detail job $DETAIL_JOB (starts per shard after the matching global shard)"
