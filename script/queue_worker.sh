#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUEUE_POP_SCRIPT="${QUEUE_POP_SCRIPT:-$ROOT_DIR/script/queue_pop_job.py}"
QUEUE_FILE="${QUEUE_FILE:-$ROOT_DIR/jobs/hmmt25_jobs.txt}"
QUEUE_HOST="${QUEUE_HOST:-}"
SSH_OPTS="${SSH_OPTS:-}"

GPU_ID="${GPU_ID:-0}"
SLEEP_SEC="${SLEEP_SEC:-30}"
EXIT_ON_EMPTY="${EXIT_ON_EMPTY:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

pop_job() {
  if [[ -n "$QUEUE_HOST" ]]; then
    ssh $SSH_OPTS "$QUEUE_HOST" "python3 '$QUEUE_POP_SCRIPT' --file '$QUEUE_FILE'"
  else
    python3 "$QUEUE_POP_SCRIPT" --file "$QUEUE_FILE"
  fi
}

while true; do
  job="$(pop_job | tr -d '\r')"
  job="${job#"${job%%[![:space:]]*}"}"
  job="${job%"${job##*[![:space:]]}"}"

  if [[ -z "$job" ]]; then
    if [[ "$EXIT_ON_EMPTY" == "1" ]]; then
      echo "Queue empty. Exiting."
      exit 0
    fi
    sleep "$SLEEP_SEC"
    continue
  fi

  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running: $job"
  if ! CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_DISABLE_COMPILE_CACHE=1 bash -lc "$job"; then
    echo "Job failed: $job" >&2
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      exit 1
    fi
  fi
done
