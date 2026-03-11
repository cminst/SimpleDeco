#!/usr/bin/env python3
import argparse
import fcntl
import os
import sys
import tempfile


def _lock_path(path: str) -> str:
    return f"{path}.lock"


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def _write_lines_atomic(path: str, lines: list[str]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".queue_tmp_", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if lines:
                handle.write("\n".join(lines))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def append_job(path: str, job: str, prepend: bool = False) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    job = job.rstrip("\n")
    if not job:
        return

    lock_path = _lock_path(path)
    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lines = _read_lines(path)
        if prepend:
            lines = [job] + lines
        else:
            lines = lines + [job]
        _write_lines_atomic(path, lines)


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
