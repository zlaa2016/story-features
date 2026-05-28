#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_494_general}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
AGE_CUTOFF="${AGE_CUTOFF:-5.0}"
TASK="${TASK:-regression}"
OUTDIR="${OUTDIR:-artifacts/finetuned_age_classifier/roberta_base_age5_regression}"
SPLIT_CSV="${SPLIT_CSV:-}"
SPLIT_COL="${SPLIT_COL:-evaluator_split}"

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --export=ALL,MODEL_NAME="${MODEL_NAME}",TASK="${TASK}",AGE_CUTOFF="${AGE_CUTOFF}",OUTDIR="${OUTDIR}"
  hpc/finetune_age_classifier.slurm
)

if [[ -n "${SPLIT_CSV}" ]]; then
  SBATCH_ARGS=(
    --account="${ACCOUNT}"
    --export=ALL,MODEL_NAME="${MODEL_NAME}",TASK="${TASK}",AGE_CUTOFF="${AGE_CUTOFF}",OUTDIR="${OUTDIR}",SPLIT_CSV="${SPLIT_CSV}",SPLIT_COL="${SPLIT_COL}"
    hpc/finetune_age_classifier.slurm
  )
fi

printf 'Submitting fine-tune job:\n'
printf '  ACCOUNT=%s\n' "${ACCOUNT}"
printf '  MODEL_NAME=%s\n' "${MODEL_NAME}"
printf '  TASK=%s\n' "${TASK}"
printf '  AGE_CUTOFF=%s\n' "${AGE_CUTOFF}"
printf '  OUTDIR=%s\n' "${OUTDIR}"
if [[ -n "${SPLIT_CSV}" ]]; then
  printf '  SPLIT_CSV=%s\n' "${SPLIT_CSV}"
  printf '  SPLIT_COL=%s\n' "${SPLIT_COL}"
else
  printf '  SPLIT_CSV=<none>\n'
fi

sbatch "${SBATCH_ARGS[@]}"
