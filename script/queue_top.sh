#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUEUE_FILE="${QUEUE_FILE:-}"
QUEUE_HOST="${QUEUE_HOST:-}"
QUEUE_STATUS_SCRIPT="${QUEUE_STATUS_SCRIPT:-$ROOT_DIR/script/queue_status.py}"
STALE_AFTER="${STALE_AFTER:-1800}"
SSH_OPTS="${SSH_OPTS:-}"
INTERVAL="${INTERVAL:-10}"
HEAD="${HEAD:-5}"
USE_CURSES="${USE_CURSES:-1}"

show_status() {
  local cmd
  cmd=(python3 "$QUEUE_STATUS_SCRIPT" --file "$QUEUE_FILE" --head "$HEAD" --stale-after "$STALE_AFTER" --watch --interval "$INTERVAL")
  if [[ "$USE_CURSES" == "0" ]]; then
    cmd+=(--no-curses)
  else
    cmd+=(--curses)
  fi
  if [[ -n "$QUEUE_HOST" ]]; then
    local cmd_str
    cmd_str=$(printf '%q ' "${cmd[@]}")
    ssh $SSH_OPTS "$QUEUE_HOST" "$cmd_str"
  else
    "${cmd[@]}"
  fi
}

if [[ -z "$QUEUE_FILE" ]]; then
  echo "Error: QUEUE_FILE must be set." >&2
  exit 1
fi

show_status
