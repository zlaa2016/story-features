#!/usr/bin/env bash
# Compression analyses on every story in the cohort (no subset, no age balancing, no completeness requirement):
#   natural-language span search, its sentence-level counterpart, the contextual and isolated vector searches,
#   and the contextual-vs-isolated shift.
#   SBATCH_ARGS="--account=<acct>" bash hpc/analysis/submit_span_search.sh
set -euo pipefail
ARGS=(${SBATCH_ARGS:-})
nl=$(sbatch --parsable "${ARGS[@]}" hpc/analysis/natural_language/detail_compression/span_search.slurm | cut -d';' -f1)
sent=$(sbatch --parsable "${ARGS[@]}" hpc/analysis/natural_language/detail_compression/span_search_sentence.slurm | cut -d';' -f1)
rep=$(sbatch --parsable "${ARGS[@]}" hpc/analysis/representation/detail_compression/span_search.slurm | cut -d';' -f1)
shift_job=$(sbatch --parsable "${ARGS[@]}" hpc/analysis/representation/detail_compression/contextualization_shift.slurm | cut -d';' -f1)
printf 'natural_language=%s\nsentence_level=%s\nrepresentation_array=%s\ncontextualization_shift=%s\n' "$nl" "$sent" "$rep" "$shift_job"
