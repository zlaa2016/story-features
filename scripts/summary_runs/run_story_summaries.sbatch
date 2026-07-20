#!/usr/bin/env bash
#SBATCH --job-name=story_summaries
#SBATCH --output=logs/summary_prompt_runs/story_summaries_%j.out
#SBATCH --error=logs/summary_prompt_runs/story_summaries_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GEN_SCRIPT="${REPO_ROOT}/scripts/generate_prompt_condition_summaries.py"

cd "${REPO_ROOT}"
mkdir -p logs/summary_prompt_runs

if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "../hartley_api_key.txt" ]; then
  export OPENAI_API_KEY="$(cat ../hartley_api_key.txt)"
fi

if [ -z "${SUMMARY_ARGS:-}" ]; then
  SUMMARY_ARGS="--limit 50 --model ${OPENAI_SUMMARY_MODEL:-gpt-4o-mini}"
fi

echo "Repo: ${REPO_ROOT}"
echo "Script: ${GEN_SCRIPT}"
echo "Args: ${SUMMARY_ARGS}"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"

python "${GEN_SCRIPT}" ${SUMMARY_ARGS}

echo "Finished: $(date)"
