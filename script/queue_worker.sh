#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUEUE_POP_SCRIPT="${QUEUE_POP_SCRIPT:-$ROOT_DIR/script/queue_pop_job.py}"
QUEUE_APPEND_SCRIPT="${QUEUE_APPEND_SCRIPT:-$ROOT_DIR/script/queue_append_job.py}"
QUEUE_FILE="${QUEUE_FILE:-$ROOT_DIR/jobs/hmmt25_jobs.txt}"
QUEUE_HOST="${QUEUE_HOST:-}"
SSH_OPTS="${SSH_OPTS:-}"
SSH_PASS="${SSH_PASS:-}"

GPU_ID="${GPU_ID:-0}"
SLEEP_SEC="${SLEEP_SEC:-30}"
EXIT_ON_EMPTY="${EXIT_ON_EMPTY:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
REQUEUE_ON_FAIL="${REQUEUE_ON_FAIL:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
FAILED_FILE="${FAILED_FILE:-$ROOT_DIR/jobs/failed_jobs.txt}"
MAX_JOBS_TO_RUN="${MAX_JOBS_TO_RUN:-0}"
RUN_COUNT=0
CURRENT_JOB=""
JOB_RUNNING=0

usage() {
  cat <<'USAGE'
Usage: script/queue_worker.sh [--ssh-pass <password>]

Options:
  --ssh-pass <password>   Use sshpass to provide the SSH password non-interactively.
                          Note: This is visible in process lists; prefer SSH keys when possible.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-pass)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --ssh-pass" >&2
        exit 1
      fi
      SSH_PASS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$SSH_PASS" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required for --ssh-pass but was not found in PATH." >&2
    exit 1
  fi
fi

requeue_current_job() {
  if [[ -n "$CURRENT_JOB" && "$JOB_RUNNING" == "1" ]]; then
    echo "Interrupted. Re-queuing: $CURRENT_JOB" >&2
    append_job "$CURRENT_JOB"
  fi
}

on_interrupt() {
  if [[ "$JOB_RUNNING" == "1" ]]; then
    echo "Interrupted. Stopping current job..." >&2
  fi
  requeue_current_job
  exit 130
}

trap on_interrupt INT
trap on_interrupt TERM

pop_job() {
  if [[ -n "$QUEUE_HOST" ]]; then
    if [[ -n "$SSH_PASS" ]]; then
      sshpass -p "$SSH_PASS" ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_POP_SCRIPT --file $QUEUE_FILE"
    else
      ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_POP_SCRIPT --file $QUEUE_FILE"
    fi
  else
    python3 "$QUEUE_POP_SCRIPT" --file "$QUEUE_FILE"
  fi
}

append_job() {
  local job_line="$1"
  if [[ -n "$QUEUE_HOST" ]]; then
    if [[ -n "$SSH_PASS" ]]; then
      printf '%s' "$job_line" | sshpass -p "$SSH_PASS" ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $QUEUE_FILE --stdin"
    else
      printf '%s' "$job_line" | ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $QUEUE_FILE --stdin"
    fi
  else
    printf '%s' "$job_line" | python3 "$QUEUE_APPEND_SCRIPT" --file "$QUEUE_FILE" --stdin
  fi
}

append_failed() {
  local job_line="$1"
  if [[ -n "$QUEUE_HOST" ]]; then
    if [[ -n "$SSH_PASS" ]]; then
      printf '%s' "$job_line" | sshpass -p "$SSH_PASS" ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $FAILED_FILE --stdin"
    else
      printf '%s' "$job_line" | ssh $SSH_OPTS "$QUEUE_HOST" "python3 $QUEUE_APPEND_SCRIPT --file $FAILED_FILE --stdin"
    fi
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
  CURRENT_JOB="$job"
  JOB_RUNNING=1
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
  JOB_RUNNING=0
  CURRENT_JOB=""

  ((RUN_COUNT += 1))
  if (( MAX_JOBS_TO_RUN > 0 && RUN_COUNT >= MAX_JOBS_TO_RUN )); then
    echo "Reached MAX_JOBS_TO_RUN=$MAX_JOBS_TO_RUN. Exiting."
    exit 0
  fi
done
