#!/usr/bin/env python3
import argparse
import fcntl
import os
import sys


def pop_job(path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        lines = handle.read().splitlines()

        job = ""
        idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            job = line
            idx = i
            break

        if idx is None:
            return ""

        del lines[idx]
        handle.seek(0)
        handle.truncate(0)
        if lines:
            handle.write("\n".join(lines))
            handle.write("\n")

    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Pop a single job from a shared queue file.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    args = parser.parse_args()

    job = pop_job(args.file)
    sys.stdout.write(job)


if __name__ == "__main__":
    main()
