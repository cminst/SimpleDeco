#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time


def _load_state(handle):
    handle.seek(0)
    raw = handle.read().strip()
    if not raw:
        return {"workers": {}}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        state = {"workers": {}}
    if not isinstance(state, dict):
        state = {"workers": {}}
    if "workers" not in state or not isinstance(state["workers"], dict):
        state["workers"] = {}
    return state


def _save_state(handle, state):
    handle.seek(0)
    handle.truncate(0)
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")


def _append_job(queue_append_script, queue_file, job, prepend):
    args = [
        sys.executable,
        queue_append_script,
        "--file",
        queue_file,
        "--stdin",
    ]
    if prepend:
        args.insert(4, "--prepend")
    subprocess.run(args, input=job, text=True, check=True)


def _reap_stale(state, now, stale_after, queue_file, queue_append_script, prepend):
    if stale_after <= 0:
        return
    workers = state.get("workers", {})
    for worker_id, info in workers.items():
        last_ping = info.get("last_ping")
        if last_ping is None:
            continue
        if now - last_ping <= stale_after:
            continue
        if info.get("status") != "running":
            continue
        if not info.get("job"):
            continue
        if info.get("requeued"):
            continue
        try:
            _append_job(queue_append_script, queue_file, info["job"], prepend=prepend)
        except subprocess.CalledProcessError:
            info["requeue_error"] = True
            continue
        info["requeued"] = True
        info["requeued_at"] = now
        info["status"] = "stale"


def _prune_workers(state, now, prune_after):
    if prune_after <= 0:
        return
    workers = state.get("workers", {})
    stale_ids = []
    for worker_id, info in workers.items():
        last_ping = info.get("last_ping")
        if last_ping is None:
            continue
        if now - last_ping > prune_after:
            stale_ids.append(worker_id)
    for worker_id in stale_ids:
        workers.pop(worker_id, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track worker status and requeue stale jobs.")
    parser.add_argument("--state-file", required=True, help="Path to worker state JSON.")
    parser.add_argument("--worker-id", help="Worker identifier.")
    parser.add_argument("--status", choices=["idle", "running", "stale", "offline"], help="Worker status.")
    parser.add_argument("--job", help="Job line.")
    parser.add_argument("--job-stdin", action="store_true", help="Read job line from stdin.")
    parser.add_argument("--progress", help="Progress string for the current job.")
    parser.add_argument("--hostname", help="Worker hostname.")
    parser.add_argument("--pid", type=int, help="Worker process id.")
    parser.add_argument("--clear", action="store_true", help="Remove worker entry from state.")
    parser.add_argument("--reap-stale", action="store_true", help="Requeue stale workers.")
    parser.add_argument("--stale-after", type=int, default=0, help="Seconds before a worker is stale.")
    parser.add_argument("--prune-after", type=int, default=0, help="Seconds before removing a worker entry.")
    parser.add_argument("--queue-file", help="Queue file path for requeue.")
    parser.add_argument("--queue-append-script", help="queue_append_job.py path for requeue.")
    parser.add_argument(
        "--prepend-stale",
        action="store_true",
        help="Prepend requeued stale jobs to the queue.",
    )
    args = parser.parse_args()

    if args.job_stdin:
        job = sys.stdin.read()
    else:
        job = args.job or ""

    parent = os.path.dirname(args.state_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.state_file, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = _load_state(handle)

        now = int(time.time())
        if args.clear and args.worker_id:
            state.get("workers", {}).pop(args.worker_id, None)
        elif args.worker_id and args.status:
            workers = state.setdefault("workers", {})
            info = workers.get(args.worker_id, {})
            info["worker_id"] = args.worker_id
            info["status"] = args.status
            info["last_ping"] = now
            if args.hostname:
                info["hostname"] = args.hostname
            if args.pid is not None:
                info["pid"] = args.pid
            if args.status == "running":
                if job:
                    info["job"] = job.rstrip("\n")
                if not info.get("job_started"):
                    info["job_started"] = now
                if args.progress is not None:
                    info["progress"] = args.progress
            else:
                info.pop("job", None)
                info.pop("job_started", None)
                info.pop("progress", None)
                info.pop("requeued", None)
                info.pop("requeued_at", None)
                info.pop("requeue_error", None)
            workers[args.worker_id] = info

        if args.reap_stale:
            if not args.queue_file or not args.queue_append_script:
                raise SystemExit("--reap-stale requires --queue-file and --queue-append-script")
            _reap_stale(
                state,
                now,
                args.stale_after,
                args.queue_file,
                args.queue_append_script,
                prepend=args.prepend_stale,
            )

        _prune_workers(state, now, args.prune_after)

        _save_state(handle, state)


if __name__ == "__main__":
    main()
