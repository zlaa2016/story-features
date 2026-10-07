#!/usr/bin/env bash
set -euo pipefail
ARGS=(${SBATCH_ARGS:-})
fit=$(sbatch --parsable "${ARGS[@]}" analysis/representation/jacobian/jacobian_lens/fit.slurm | cut -d';' -f1)
merge=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$fit" analysis/representation/jacobian/jacobian_lens/merge.slurm | cut -d';' -f1)
apply=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$merge" analysis/representation/jacobian/jacobian_lens/apply.slurm | cut -d';' -f1)
analysis=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$apply" analysis/representation/jacobian/jacobian_lens/analyze.slurm | cut -d';' -f1)
intervention=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$analysis" analysis/representation/jacobian/jacobian_lens/intervene.slurm | cut -d';' -f1)
printf 'jlens_fit=%s\njlens_merge=%s\njlens_apply=%s\njlens_analysis=%s\njlens_intervention=%s\n' "$fit" "$merge" "$apply" "$analysis" "$intervention"
