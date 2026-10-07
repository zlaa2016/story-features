#!/usr/bin/env bash

set -euo pipefail
: "${MODEL_KEY:=gpt_oss_20b}"
: "${SBATCH_ARGS:=--account=torch_pr_494_general}"

prepare_job=$(MODEL_KEY="$MODEL_KEY" sbatch --parsable $SBATCH_ARGS hpc/analysis/representation/geometry/compression_geometry/prepare.slurmrepresentation/geometry/compression_geometry/prepare.slurm | cut -d';' -f1)
align_job=$(MODEL_KEY="$MODEL_KEY" sbatch --parsable $SBATCH_ARGS --dependency="afterok:${prepare_job}" hpc/analysis/representation/geometry/compression_geometry/align.slurmrepresentation/geometry/compression_geometry/align.slurm | cut -d';' -f1)
analysis_job=$(MODEL_KEY="$MODEL_KEY" sbatch --parsable $SBATCH_ARGS --dependency="afterok:${align_job}" hpc/analysis/representation/geometry/compression_geometry/analyze.slurmrepresentation/geometry/compression_geometry/analyze.slurm | cut -d';' -f1)

printf 'model=%s\nprepare=%s\nalign=%s\nanalyze=%s\n' "$MODEL_KEY" "$prepare_job" "$align_job" "$analysis_job"
