#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUEUE_POP_SCRIPT="${QUEUE_POP_SCRIPT:-$ROOT_DIR/script/queue_pop_job.py}"
QUEUE_APPEND_SCRIPT="${QUEUE_APPEND_SCRIPT:-$ROOT_DIR/script/queue_append_job.py}"
QUEUE_FILE="${QUEUE_FILE:-$ROOT_DIR/jobs/hmmt25_jobs.txt}"
QUEUE_HOST="${QUEUE_HOST:-}"
SSH_OPTS="${SSH_OPTS:-}"

GPU_ID="${GPU_ID:-0}"
SLEEP_SEC="${SLEEP_SEC:-30}"
EXIT_ON_EMPTY="${EXIT_ON_EMPTY:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
REQUEUE_ON_FAIL="${REQUEUE_ON_FAIL:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
FAILED_FILE="${FAILED_FILE:-$ROOT_DIR/jobs/failed_jobs.txt}"

pop_job() {
  if [[ -n "$QUEUE_HOST" ]]; then
    ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_POP_SCRIPT --file $QUEUE_FILE"
  else
    python3 "$QUEUE_POP_SCRIPT" --file "$QUEUE_FILE"
  fi
}

append_job() {
  local job_line="$1"
  if [[ -n "$QUEUE_HOST" ]]; then
    printf '%s' "$job_line" | ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $QUEUE_FILE --stdin"
  else
    printf '%s' "$job_line" | python3 "$QUEUE_APPEND_SCRIPT" --file "$QUEUE_FILE" --stdin
  fi
}

append_failed() {
  local job_line="$1"
  if [[ -n "$QUEUE_HOST" ]]; then
    printf '%s' "$job_line" | ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $FAILED_FILE --stdin"
  else
    printf '%s' "$job_line" | python3 "$QUEUE_APPEND_SCRIPT" --file "$FAILED_FILE" --stdin
  fi
}

parse_retry() {
  local job_line="$1"
  local retries=0
  if [[ "$job_line" =~ \#\ retry=([0-9]+) ]]; then
    retries="${BASH_REMATCH[1]}"
  fi
  echo "$retries"
}

strip_retry() {
  local job_line="$1"
  job_line="${job_line%%# retry=*}"
  job_line="${job_line%"${job_line##*[![:space:]]}"}"
  echo "$job_line"
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
  if ! CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_DISABLE_COMPILE_CACHE=1 bash -lc "cd \"$ROOT_DIR\" && $job"; then
    echo "Job failed: $job" >&2
    if [[ "$REQUEUE_ON_FAIL" == "1" ]]; then
      retries="$(parse_retry "$job")"
      if (( retries < MAX_RETRIES )); then
        base_job="$(strip_retry "$job")"
        requeued="${base_job} # retry=$((retries + 1))"
        echo "Re-queuing (attempt $((retries + 1))/$MAX_RETRIES): $base_job"
        append_job "$requeued"
      else
        echo "Max retries reached. Recording failure."
        append_failed "$job"
      fi
    fi
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      exit 1
    fi
  fi
done
