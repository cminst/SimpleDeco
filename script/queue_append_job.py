#!/usr/bin/env python3
import argparse
import fcntl
import os
import sys


def append_job(path: str, job: str, prepend: bool = False) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    job = job.rstrip("\n")
    if not job:
        return

    if prepend:
        mode = "r+" if os.path.exists(path) else "w+"
    else:
        mode = "a+"

    with open(path, mode) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if prepend:
            handle.seek(0)
            existing = handle.read().rstrip("\n")
            handle.seek(0)
            handle.truncate(0)
            handle.write(job)
            handle.write("\n")
            if existing:
                handle.write(existing)
                handle.write("\n")
        else:
            handle.seek(0, os.SEEK_END)
            if handle.tell() > 0:
                handle.write("\n")
            handle.write(job)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a single job to a shared queue file.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    parser.add_argument("--prepend", action="store_true", help="Prepend the job to the front of the queue.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", help="Job line to append.")
    group.add_argument("--stdin", action="store_true", help="Read job line from stdin.")
    args = parser.parse_args()

    if args.stdin:
        job = sys.stdin.read()
    else:
        job = args.job or ""

    append_job(args.file, job, prepend=args.prepend)


if __name__ == "__main__":
    main()
