#!/usr/bin/env python3
import argparse
import fcntl
import os
import sys


def append_job(path: str, job: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    job = job.rstrip("\n")
    if not job:
        return

    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.write("\n")
        handle.write(job)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a single job to a shared queue file.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", help="Job line to append.")
    group.add_argument("--stdin", action="store_true", help="Read job line from stdin.")
    args = parser.parse_args()

    if args.stdin:
        job = sys.stdin.read()
    else:
        job = args.job or ""

    append_job(args.file, job)


if __name__ == "__main__":
    main()
