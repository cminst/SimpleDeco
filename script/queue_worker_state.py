#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from queue_backend import reap_stale_jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy compatibility wrapper. Reaps stale leases in the durable queue."
    )
    parser.add_argument("--file", help="Queue inbox file.")
    parser.add_argument("--queue-file", help="Queue inbox file.")
    parser.add_argument("--stale-after", type=int, default=0, help="Seconds before a running lease is stale.")
    parser.add_argument("--prepend-stale", action="store_true", help="Requeue stale jobs at the front.")
    parser.add_argument("--json", action="store_true", help="Print requeued job records.")
    args, unknown = parser.parse_known_args()

    del unknown
    queue_file = args.queue_file or args.file
    if not queue_file:
        parser.error("--queue-file is required.")

    requeued = reap_stale_jobs(
        queue_file,
        stale_after=args.stale_after,
        prepend=args.prepend_stale,
    )
    if args.json:
        sys.stdout.write(json.dumps(requeued, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(f"{len(requeued)}\n")


if __name__ == "__main__":
    main()
