#!/usr/bin/env python3
import argparse
import json
import os
import time


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
    parser.add_argument("--state-file", help="Worker state JSON file.")
    parser.add_argument("--stale-after", type=int, default=0, help="Seconds before worker is stale.")
    args = parser.parse_args()

    jobs = load_jobs(args.file)
    print(f"Queue file: {args.file}")
    print(f"Remaining jobs: {len(jobs)}")
    if args.head > 0 and jobs:
        print("Next jobs:")
        for job in jobs[: args.head]:
            print(f"- {job}")

    if args.state_file and os.path.exists(args.state_file):
        with open(args.state_file, "r") as handle:
            try:
                state = json.load(handle)
            except json.JSONDecodeError:
                state = {}
        workers = state.get("workers", {})
        if workers:
            now = time.time()
            print()
            print("Workers:")
            for worker_id in sorted(workers):
                info = workers[worker_id]
                last_ping = info.get("last_ping")
                age = None if last_ping is None else int(now - last_ping)
                status = info.get("status", "unknown")
                if age is not None and args.stale_after > 0 and age > args.stale_after:
                    status = "stale"
                age_str = "unknown" if age is None else f"{age}s"
                job = info.get("job")
                if job and len(job) > 120:
                    job = job[:117] + "..."
                if job:
                    print(f"- {worker_id} | {status} | last ping {age_str} | job: {job}")
                else:
                    print(f"- {worker_id} | {status} | last ping {age_str}")


if __name__ == "__main__":
    main()
