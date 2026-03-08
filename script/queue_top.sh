#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUEUE_FILE="${QUEUE_FILE:-$ROOT_DIR/jobs/hmmt25_jobs.txt}"
QUEUE_HOST="${QUEUE_HOST:-}"
QUEUE_STATUS_SCRIPT="${QUEUE_STATUS_SCRIPT:-$ROOT_DIR/script/queue_status.py}"
SSH_OPTS="${SSH_OPTS:-}"
INTERVAL="${INTERVAL:-10}"
HEAD="${HEAD:-5}"

show_status() {
  if [[ -n "$QUEUE_HOST" ]]; then
    ssh $SSH_OPTS "$QUEUE_HOST" "python3 '$QUEUE_STATUS_SCRIPT' --file '$QUEUE_FILE' --head '$HEAD'"
  else
    python3 "$QUEUE_STATUS_SCRIPT" --file "$QUEUE_FILE" --head "$HEAD"
  fi
}

while true; do
  clear
  date +"%Y-%m-%d %H:%M:%S"
  echo
  show_status
  sleep "$INTERVAL"
done
