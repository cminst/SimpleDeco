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


def pop_job(path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lock_path = _lock_path(path)
    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lines = _read_lines(path)

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
        _write_lines_atomic(path, lines)

    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Pop a single job from a shared queue file.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    args = parser.parse_args()

    job = pop_job(args.file)
    sys.stdout.write(job)


if __name__ == "__main__":
    main()
