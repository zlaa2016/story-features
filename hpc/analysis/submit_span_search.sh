#!/usr/bin/env bash
# Select the 3-condition subset, then run the span searches and the contextual-vs-isolated shift on it.
#   SBATCH_ARGS="--account=<acct>" bash hpc/analysis/submit_span_search.sh
set -euo pipefail
ARGS=(${SBATCH_ARGS:-})
subset=$(sbatch --parsable "${ARGS[@]}" hpc/features/summaries/dataset_index/select_flat_subset.slurm | cut -d';' -f1)
nl=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$subset" hpc/analysis/natural_language/detail_compression/span_search.slurm | cut -d';' -f1)
rep=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$subset" hpc/analysis/representation/detail_compression/span_search.slurm | cut -d';' -f1)
shift=$(sbatch --parsable "${ARGS[@]}" --dependency="afterok:$subset" hpc/analysis/representation/detail_compression/contextualization_shift.slurm | cut -d';' -f1)
printf 'subset=%s\nnatural_language=%s\nrepresentation_array=%s\ncontextualization_shift=%s\n' "$subset" "$nl" "$rep" "$shift"
