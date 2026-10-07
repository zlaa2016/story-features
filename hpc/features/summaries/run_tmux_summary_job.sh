#!/usr/bin/env bash
set -euo pipefail

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GEN_SCRIPT="${REPO_ROOT}/features/summaries/generate_prompt_condition_summaries.py"
SESSION_NAME="${SESSION_NAME:-story_summaries}"
LOG_DIR="${REPO_ROOT}/logs/summary_prompt_runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/tmux_${SESSION_NAME}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=(--limit 10)
fi

printf -v REPO_Q "%q" "${REPO_ROOT}"
printf -v LOG_Q "%q" "${LOG_FILE}"
printf -v PY_CMD_Q "%q " python "${GEN_SCRIPT}" "${ARGS[@]}"

RUN_CMD="cd ${REPO_Q} && ${PY_CMD_Q} 2>&1 | tee ${LOG_Q}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Detach with: Ctrl-b then d"
