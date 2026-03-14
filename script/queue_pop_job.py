#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from queue_backend import claim_job


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claim one job from the durable queue. This is a manual helper; queue_worker.py should be preferred."
    )
    parser.add_argument("--file", required=True, help="Queue inbox file.")
    parser.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}-{os.getpid()}",
        help="Worker identifier recorded on the claim.",
    )
    parser.add_argument("--hostname", default=socket.gethostname(), help="Hostname to record.")
    parser.add_argument("--json", action="store_true", help="Print the full claimed record.")
    args = parser.parse_args()

    record = claim_job(args.file, worker_id=args.worker_id, hostname=args.hostname)
    if args.json:
        sys.stdout.write(json.dumps(record, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    elif record:
        sys.stdout.write(record["job"])


if __name__ == "__main__":
    main()
