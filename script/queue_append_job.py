#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from queue_backend import enqueue_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Add one job to the durable queue.")
    parser.add_argument("--file", required=True, help="Queue inbox file.")
    parser.add_argument("--prepend", action="store_true", help="Insert at the front of the queue.")
    parser.add_argument("--json", action="store_true", help="Print the created job record.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", help="Job line to append.")
    group.add_argument("--stdin", action="store_true", help="Read the job from stdin.")
    args = parser.parse_args()

    job = sys.stdin.read() if args.stdin else (args.job or "")
    record = enqueue_job(args.file, job, prepend=args.prepend)
    if args.json:
        sys.stdout.write(json.dumps(record, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    elif record:
        sys.stdout.write(record["job_id"] + "\n")


if __name__ == "__main__":
    main()
