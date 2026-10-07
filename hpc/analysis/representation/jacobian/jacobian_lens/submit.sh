#!/usr/bin/env bash
set -euo pipefail
ARGS=(${SBATCH_ARGS:-})
fit=$(sbatch --parsable "${ARGS[@]}" hpc/analysis/representation/jacobian/jacobian_lens/fit.slurmrepresentation/jacobian/jacobian_lens/fit.slurm | cut -d';' -f1)
merge=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$fit" hpc/analysis/representation/jacobian/jacobian_lens/merge.slurmrepresentation/jacobian/jacobian_lens/merge.slurm | cut -d';' -f1)
apply=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$merge" hpc/analysis/representation/jacobian/jacobian_lens/apply.slurmrepresentation/jacobian/jacobian_lens/apply.slurm | cut -d';' -f1)
analysis=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$apply" hpc/analysis/representation/jacobian/jacobian_lens/analyze.slurmrepresentation/jacobian/jacobian_lens/analyze.slurm | cut -d';' -f1)
intervention=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$analysis" hpc/analysis/representation/jacobian/jacobian_lens/intervene.slurmrepresentation/jacobian/jacobian_lens/intervene.slurm | cut -d';' -f1)
printf 'jlens_fit=%s\njlens_merge=%s\njlens_apply=%s\njlens_analysis=%s\njlens_intervention=%s\n' "$fit" "$merge" "$apply" "$analysis" "$intervention"
