#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

from queue_backend import collect_status, reap_stale_jobs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Host-side watchdog that requeues stale leases and prints queue counts."
    )
    parser.add_argument("--queue-file", default=_env_str("QUEUE_FILE", ""))
    parser.add_argument("--stale-after", type=int, default=_env_int("STALE_AFTER", 180))
    parser.add_argument("--interval", type=float, default=_env_int("WATCHDOG_INTERVAL", 30))
    parser.add_argument("--head", type=int, default=_env_int("HEAD", 5))
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--exit-on-empty",
        action="store_true",
        help="Exit once there are no pending or running jobs.",
    )
    args = parser.parse_args()

    if not args.queue_file:
        parser.error("--queue-file is required (or set QUEUE_FILE).")

    while True:
        requeued = reap_stale_jobs(args.queue_file, stale_after=args.stale_after, prepend=True)
        status = collect_status(args.queue_file, stale_after=args.stale_after, head=args.head)
        running = sum(1 for row in status["workers"] if not row.get("stale"))
        stale = sum(1 for row in status["workers"] if row.get("stale"))
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now_str}] pending={status['remaining']} completed={status['completed']} failed={status['failed']} "
            f"running={running} stale={stale} requeued={len(requeued)}"
        )
        if args.once:
            break
        if args.exit_on_empty and status["remaining"] == 0 and not status["workers"]:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
