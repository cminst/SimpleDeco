#!/usr/bin/env python3
import argparse
import os


def load_jobs(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r") as handle:
        lines = handle.read().splitlines()
    jobs = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        jobs.append(line)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Show queue status.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    parser.add_argument("--head", type=int, default=5, help="Show next N jobs.")
    args = parser.parse_args()

    jobs = load_jobs(args.file)
    print(f"Queue file: {args.file}")
    print(f"Remaining jobs: {len(jobs)}")
    if args.head > 0 and jobs:
        print("Next jobs:")
        for job in jobs[: args.head]:
            print(f"- {job}")


if __name__ == "__main__":
    main()
