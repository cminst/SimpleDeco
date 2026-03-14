#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _clean_queue_name(queue_file: str) -> str:
    base = os.path.basename(queue_file)
    if "." in base:
        base = base.rsplit(".", 1)[0]
    if base.endswith("_jobs"):
        base = base[: -len("_jobs")]
    return base or "queue"


def default_queue_dir(queue_file: str) -> str:
    parent = os.path.dirname(queue_file) or "."
    return os.path.join(parent, f"{_clean_queue_name(queue_file)}_queue")


@dataclass(frozen=True)
class QueuePaths:
    queue_file: Path
    queue_dir: Path
    lock_file: Path
    meta_file: Path
    pending_dir: Path
    running_dir: Path
    completed_dir: Path
    failed_dir: Path

    @classmethod
    def from_queue_file(cls, queue_file: str, queue_dir: str | None = None) -> "QueuePaths":
        queue_file_path = Path(queue_file).expanduser()
        queue_dir_path = Path(queue_dir or default_queue_dir(str(queue_file_path))).expanduser()
        return cls(
            queue_file=queue_file_path,
            queue_dir=queue_dir_path,
            lock_file=queue_dir_path / ".lock",
            meta_file=queue_dir_path / "meta.json",
            pending_dir=queue_dir_path / "pending",
            running_dir=queue_dir_path / "running",
            completed_dir=queue_dir_path / "completed",
            failed_dir=queue_dir_path / "failed",
        )


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".queue_tmp_", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_empty(path: Path) -> None:
    _write_atomic(path, "")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _load_meta(paths: QueuePaths) -> dict[str, int]:
    if not paths.meta_file.exists():
        return {"next_back_seq": 1, "next_front_seq": 0}
    payload = _load_json(paths.meta_file)
    next_back_seq = int(payload.get("next_back_seq", 1))
    next_front_seq = int(payload.get("next_front_seq", 0))
    return {"next_back_seq": next_back_seq, "next_front_seq": next_front_seq}


def _save_meta(paths: QueuePaths, meta: dict[str, int]) -> None:
    _write_json(paths.meta_file, meta)


def _ensure_layout(paths: QueuePaths) -> None:
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    paths.pending_dir.mkdir(parents=True, exist_ok=True)
    paths.running_dir.mkdir(parents=True, exist_ok=True)
    paths.completed_dir.mkdir(parents=True, exist_ok=True)
    paths.failed_dir.mkdir(parents=True, exist_ok=True)
    if not paths.meta_file.exists():
        _save_meta(paths, {"next_back_seq": 1, "next_front_seq": 0})


@contextmanager
def queue_lock(paths: QueuePaths):
    _ensure_layout(paths)
    with open(paths.lock_file, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _pending_name(order: int, job_id: str) -> str:
    return f"{order:+020d}__{job_id}.json"


def _running_name(order: int, claimed_at_ms: int, job_id: str, lease_id: str) -> str:
    return f"{order:+020d}__{claimed_at_ms:013d}__{job_id}__{lease_id}.json"


def _parse_pending_name(path: Path) -> tuple[int, str]:
    order_text, job_id_json = path.name.split("__", 1)
    job_id = job_id_json.rsplit(".", 1)[0]
    return int(order_text), job_id


def _parse_running_name(path: Path) -> tuple[int, int, str, str]:
    order_text, claimed_at_text, job_id, lease_json = path.name.split("__", 3)
    lease_id = lease_json.rsplit(".", 1)[0]
    return int(order_text), int(claimed_at_text), job_id, lease_id


def _load_submission_lines(queue_file: Path) -> list[str]:
    if not queue_file.exists():
        return []
    with open(queue_file, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def _next_order(meta: dict[str, int], prepend: bool) -> int:
    if prepend:
        order = meta["next_front_seq"]
        meta["next_front_seq"] -= 1
        return order
    order = meta["next_back_seq"]
    meta["next_back_seq"] += 1
    return order


def _new_job_record(job: str, job_id: str, order: int, now: float, source: str) -> dict[str, Any]:
    return {
        "job": job,
        "job_id": job_id,
        "order": order,
        "progress": "",
        "retries": 0,
        "source": source,
        "status": "pending",
        "stale_requeues": 0,
        "submitted_at": now,
        "updated_at": now,
    }


def _enqueue_jobs_locked(
    paths: QueuePaths,
    jobs: list[str],
    prepend: bool,
    source: str,
    now: float | None = None,
) -> list[dict[str, Any]]:
    normalized = [job.rstrip("\n") for job in jobs if job.rstrip("\n").strip()]
    if not normalized:
        return []
    meta = _load_meta(paths)
    planned: list[tuple[int, str, dict[str, Any]]] = []
    created: list[dict[str, Any]] = []
    current_time = now if now is not None else time.time()
    for job in normalized:
        order = _next_order(meta, prepend=prepend)
        job_id = uuid.uuid4().hex
        record = _new_job_record(job, job_id, order, current_time, source=source)
        planned.append((order, job_id, record))
        created.append(record)
    _save_meta(paths, meta)
    for order, job_id, record in planned:
        _write_json(paths.pending_dir / _pending_name(order, job_id), record)
    return created


def sync_submission_file(paths: QueuePaths, now: float | None = None) -> int:
    with queue_lock(paths):
        return _sync_submission_file_locked(paths, now=now)


def _sync_submission_file_locked(paths: QueuePaths, now: float | None = None) -> int:
    jobs = _load_submission_lines(paths.queue_file)
    if not jobs:
        return 0
    _enqueue_jobs_locked(paths, jobs, prepend=False, source="queue_file", now=now)
    _write_empty(paths.queue_file)
    return len(jobs)


def enqueue_job(
    queue_file: str,
    job: str,
    prepend: bool = False,
    queue_dir: str | None = None,
) -> dict[str, Any] | None:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        created = _enqueue_jobs_locked(paths, [job], prepend=prepend, source="manual")
    return created[0] if created else None


def _pending_paths(paths: QueuePaths) -> list[Path]:
    items = [path for path in paths.pending_dir.iterdir() if path.suffix == ".json"]
    return sorted(items, key=lambda path: _parse_pending_name(path)[0])


def _running_paths(paths: QueuePaths) -> list[Path]:
    items = [path for path in paths.running_dir.iterdir() if path.suffix == ".json"]
    return sorted(items, key=lambda path: _parse_running_name(path)[1])


def _find_running_path(paths: QueuePaths, job_id: str, lease_id: str) -> Path | None:
    suffix = f"__{job_id}__{lease_id}.json"
    for path in paths.running_dir.iterdir():
        if path.name.endswith(suffix):
            return path
    return None


def claim_job(
    queue_file: str,
    worker_id: str,
    hostname: str | None = None,
    queue_dir: str | None = None,
    sync: bool = True,
) -> dict[str, Any] | None:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        if sync:
            _sync_submission_file_locked(paths)
        pending = _pending_paths(paths)
        if not pending:
            return None
        path = pending[0]
        order, job_id = _parse_pending_name(path)
        record = _load_json(path)
        lease_id = uuid.uuid4().hex
        claimed_at = time.time()
        claimed_at_ms = int(claimed_at * 1000)
        running_path = paths.running_dir / _running_name(order, claimed_at_ms, job_id, lease_id)
        os.replace(path, running_path)
        _fsync_dir(paths.pending_dir)
        _fsync_dir(paths.running_dir)

        record["status"] = "running"
        record["worker_id"] = worker_id
        record["hostname"] = hostname or socket.gethostname()
        record["lease_id"] = lease_id
        record["claimed_at"] = claimed_at
        record["heartbeat_at"] = claimed_at
        record["progress"] = ""
        record["updated_at"] = claimed_at
        record["attempt"] = int(record.get("attempt", 0)) + 1
        _write_json(running_path, record)
        return record


def heartbeat_job(
    queue_file: str,
    job_id: str,
    lease_id: str,
    worker_id: str,
    progress: str | None = None,
    hostname: str | None = None,
    queue_dir: str | None = None,
) -> bool:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        path = _find_running_path(paths, job_id, lease_id)
        if path is None:
            return False
        record = _load_json(path)
        if record.get("worker_id") not in {None, worker_id}:
            return False
        now = time.time()
        record["worker_id"] = worker_id
        record["hostname"] = hostname or record.get("hostname") or socket.gethostname()
        record["heartbeat_at"] = now
        record["updated_at"] = now
        if progress is not None:
            record["progress"] = progress
        _write_json(path, record)
        return True


def _move_running_to_pending(
    paths: QueuePaths,
    running_path: Path,
    record: dict[str, Any],
    prepend: bool,
    now: float,
    stale_requeue: bool,
    last_error: str | None = None,
) -> dict[str, Any]:
    meta = _load_meta(paths)
    order = _next_order(meta, prepend=prepend)
    _save_meta(paths, meta)
    job_id = record["job_id"]
    pending_path = paths.pending_dir / _pending_name(order, job_id)
    os.replace(running_path, pending_path)
    _fsync_dir(paths.running_dir)
    _fsync_dir(paths.pending_dir)
    record["status"] = "pending"
    record["order"] = order
    record["updated_at"] = now
    record.pop("worker_id", None)
    record.pop("hostname", None)
    record.pop("lease_id", None)
    record.pop("claimed_at", None)
    record.pop("heartbeat_at", None)
    record["progress"] = ""
    if last_error is not None:
        record["last_error"] = last_error
        record["last_failed_at"] = now
    if stale_requeue:
        record["stale_requeues"] = int(record.get("stale_requeues", 0)) + 1
        record["last_requeued_at"] = now
    _write_json(pending_path, record)
    return record


def release_job(
    queue_file: str,
    job_id: str,
    lease_id: str,
    worker_id: str,
    prepend: bool = False,
    queue_dir: str | None = None,
) -> bool:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        path = _find_running_path(paths, job_id, lease_id)
        if path is None:
            return False
        record = _load_json(path)
        if record.get("worker_id") not in {None, worker_id}:
            return False
        _move_running_to_pending(
            paths,
            path,
            record,
            prepend=prepend,
            now=time.time(),
            stale_requeue=False,
        )
        return True


def fail_job(
    queue_file: str,
    job_id: str,
    lease_id: str,
    worker_id: str,
    max_retries: int,
    prepend: bool = False,
    error: str | None = None,
    queue_dir: str | None = None,
) -> dict[str, Any]:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        path = _find_running_path(paths, job_id, lease_id)
        if path is None:
            return {"ok": False, "action": "missing"}
        record = _load_json(path)
        if record.get("worker_id") not in {None, worker_id}:
            return {"ok": False, "action": "lost_lease"}
        now = time.time()
        retries = int(record.get("retries", 0))
        if retries < max_retries:
            record["retries"] = retries + 1
            updated = _move_running_to_pending(
                paths,
                path,
                record,
                prepend=prepend,
                now=now,
                stale_requeue=False,
                last_error=error,
            )
            return {"ok": True, "action": "retried", "record": updated}

        failed_path = paths.failed_dir / f"{record['job_id']}.json"
        os.replace(path, failed_path)
        _fsync_dir(paths.running_dir)
        _fsync_dir(paths.failed_dir)
        record["status"] = "failed"
        record["failed_at"] = now
        record["updated_at"] = now
        record["last_error"] = error or record.get("last_error", "")
        _write_json(failed_path, record)
        return {"ok": True, "action": "failed", "record": record}


def complete_job(
    queue_file: str,
    job_id: str,
    lease_id: str,
    worker_id: str,
    queue_dir: str | None = None,
) -> bool:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        path = _find_running_path(paths, job_id, lease_id)
        if path is None:
            return False
        record = _load_json(path)
        if record.get("worker_id") not in {None, worker_id}:
            return False
        completed_path = paths.completed_dir / f"{record['job_id']}.json"
        os.replace(path, completed_path)
        _fsync_dir(paths.running_dir)
        _fsync_dir(paths.completed_dir)
        now = time.time()
        record["status"] = "completed"
        record["completed_at"] = now
        record["updated_at"] = now
        _write_json(completed_path, record)
        return True


def reap_stale_jobs(
    queue_file: str,
    stale_after: int,
    prepend: bool = True,
    queue_dir: str | None = None,
    sync: bool = True,
) -> list[dict[str, Any]]:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    if stale_after <= 0:
        return []
    requeued: list[dict[str, Any]] = []
    with queue_lock(paths):
        if sync:
            _sync_submission_file_locked(paths)
        now = time.time()
        for running_path in _running_paths(paths):
            _, claimed_at_ms, _job_id, _lease_id = _parse_running_name(running_path)
            record = _load_json(running_path)
            last_seen = float(
                record.get("heartbeat_at")
                or record.get("claimed_at")
                or (claimed_at_ms / 1000.0)
            )
            if now - last_seen <= stale_after:
                continue
            updated = _move_running_to_pending(
                paths,
                running_path,
                record,
                prepend=prepend,
                now=now,
                stale_requeue=True,
            )
            requeued.append(updated)
    return requeued


def _job_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": record.get("job_id", ""),
        "job": record.get("job", ""),
        "progress": record.get("progress", ""),
        "retries": int(record.get("retries", 0)),
        "status": record.get("status", ""),
        "worker_id": record.get("worker_id", ""),
        "hostname": record.get("hostname", ""),
        "heartbeat_at": record.get("heartbeat_at"),
        "claimed_at": record.get("claimed_at"),
    }


def collect_status(
    queue_file: str,
    queue_dir: str | None = None,
    stale_after: int = 0,
    head: int = 5,
    sync: bool = False,
) -> dict[str, Any]:
    paths = QueuePaths.from_queue_file(queue_file, queue_dir=queue_dir)
    with queue_lock(paths):
        if sync:
            _sync_submission_file_locked(paths)
        pending_paths = _pending_paths(paths)
        inbox_jobs = [] if sync else _load_submission_lines(paths.queue_file)
        running_paths = _running_paths(paths)
        completed_paths = [path for path in paths.completed_dir.iterdir() if path.suffix == ".json"]
        failed_paths = [path for path in paths.failed_dir.iterdir() if path.suffix == ".json"]

        pending_jobs = []
        for path in pending_paths[: max(0, head)]:
            record = _load_json(path)
            pending_jobs.append(_job_summary(record))
        if len(pending_jobs) < head:
            for line in inbox_jobs[: max(0, head - len(pending_jobs))]:
                pending_jobs.append(
                    {
                        "job_id": "",
                        "job": line,
                        "progress": "",
                        "retries": 0,
                        "status": "pending",
                        "worker_id": "",
                        "hostname": "",
                        "heartbeat_at": None,
                        "claimed_at": None,
                    }
                )

        now = time.time()
        running_jobs = []
        for path in running_paths:
            _order, claimed_at_ms, _job_id, _lease_id = _parse_running_name(path)
            record = _load_json(path)
            last_seen = float(
                record.get("heartbeat_at")
                or record.get("claimed_at")
                or (claimed_at_ms / 1000.0)
            )
            summary = _job_summary(record)
            summary["age_seconds"] = int(max(0, now - last_seen))
            summary["stale"] = bool(stale_after > 0 and now - last_seen > stale_after)
            running_jobs.append(summary)

        return {
            "queue_file": str(paths.queue_file),
            "queue_dir": str(paths.queue_dir),
            "remaining": len(pending_paths) + len(inbox_jobs),
            "completed": len(completed_paths),
            "failed": len(failed_paths),
            "jobs": pending_jobs,
            "workers": running_jobs,
        }


def _print_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable queue backend.")
    parser.add_argument("--queue-file", required=True, help="Queue inbox file.")
    parser.add_argument("--queue-dir", help="Override durable queue directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue_parser = subparsers.add_parser("enqueue", help="Add a job to the durable queue.")
    enqueue_parser.add_argument("--prepend", action="store_true")
    enqueue_group = enqueue_parser.add_mutually_exclusive_group(required=True)
    enqueue_group.add_argument("--job")
    enqueue_group.add_argument("--stdin", action="store_true")
    enqueue_parser.add_argument("--json", action="store_true")

    claim_parser = subparsers.add_parser("claim", help="Claim one pending job.")
    claim_parser.add_argument("--worker-id", required=True)
    claim_parser.add_argument("--hostname", default=socket.gethostname())
    claim_parser.add_argument("--json", action="store_true")

    heartbeat_parser = subparsers.add_parser("heartbeat", help="Refresh a running lease.")
    heartbeat_parser.add_argument("--job-id", required=True)
    heartbeat_parser.add_argument("--lease-id", required=True)
    heartbeat_parser.add_argument("--worker-id", required=True)
    heartbeat_parser.add_argument("--hostname", default=socket.gethostname())
    heartbeat_parser.add_argument("--progress")

    complete_parser = subparsers.add_parser("complete", help="Mark a lease completed.")
    complete_parser.add_argument("--job-id", required=True)
    complete_parser.add_argument("--lease-id", required=True)
    complete_parser.add_argument("--worker-id", required=True)

    release_parser = subparsers.add_parser("release", help="Return a lease to pending.")
    release_parser.add_argument("--job-id", required=True)
    release_parser.add_argument("--lease-id", required=True)
    release_parser.add_argument("--worker-id", required=True)
    release_parser.add_argument("--prepend", action="store_true")

    fail_parser = subparsers.add_parser("fail", help="Retry or fail a lease.")
    fail_parser.add_argument("--job-id", required=True)
    fail_parser.add_argument("--lease-id", required=True)
    fail_parser.add_argument("--worker-id", required=True)
    fail_parser.add_argument("--max-retries", type=int, default=0)
    fail_parser.add_argument("--prepend", action="store_true")
    fail_parser.add_argument("--error")
    fail_parser.add_argument("--json", action="store_true")

    reap_parser = subparsers.add_parser("reap", help="Return stale leases to pending.")
    reap_parser.add_argument("--stale-after", type=int, required=True)
    reap_parser.add_argument("--prepend", action="store_true")
    reap_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show queue counts.")
    status_parser.add_argument("--stale-after", type=int, default=0)
    status_parser.add_argument("--head", type=int, default=5)
    status_parser.add_argument("--json", action="store_true")

    sync_parser = subparsers.add_parser("sync", help="Import jobs from the inbox file.")
    sync_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "enqueue":
        job_text = sys.stdin.read() if args.stdin else (args.job or "")
        created = enqueue_job(args.queue_file, job_text, prepend=args.prepend, queue_dir=args.queue_dir)
        if args.json:
            _print_json(created)
        elif created:
            sys.stdout.write(created["job_id"] + "\n")
        return

    if args.command == "claim":
        record = claim_job(
            args.queue_file,
            worker_id=args.worker_id,
            hostname=args.hostname,
            queue_dir=args.queue_dir,
        )
        if args.json:
            _print_json(record)
        elif record:
            sys.stdout.write(record["job"] + "\n")
        return

    if args.command == "heartbeat":
        ok = heartbeat_job(
            args.queue_file,
            job_id=args.job_id,
            lease_id=args.lease_id,
            worker_id=args.worker_id,
            hostname=args.hostname,
            progress=args.progress,
            queue_dir=args.queue_dir,
        )
        raise SystemExit(0 if ok else 1)

    if args.command == "complete":
        ok = complete_job(
            args.queue_file,
            job_id=args.job_id,
            lease_id=args.lease_id,
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
        )
        raise SystemExit(0 if ok else 1)

    if args.command == "release":
        ok = release_job(
            args.queue_file,
            job_id=args.job_id,
            lease_id=args.lease_id,
            worker_id=args.worker_id,
            prepend=args.prepend,
            queue_dir=args.queue_dir,
        )
        raise SystemExit(0 if ok else 1)

    if args.command == "fail":
        result = fail_job(
            args.queue_file,
            job_id=args.job_id,
            lease_id=args.lease_id,
            worker_id=args.worker_id,
            max_retries=args.max_retries,
            prepend=args.prepend,
            error=args.error,
            queue_dir=args.queue_dir,
        )
        if args.json:
            _print_json(result)
        raise SystemExit(0 if result.get("ok") else 1)

    if args.command == "reap":
        requeued = reap_stale_jobs(
            args.queue_file,
            stale_after=args.stale_after,
            prepend=args.prepend,
            queue_dir=args.queue_dir,
        )
        if args.json:
            _print_json(requeued)
        else:
            sys.stdout.write(f"{len(requeued)}\n")
        return

    if args.command == "status":
        status = collect_status(
            args.queue_file,
            queue_dir=args.queue_dir,
            stale_after=args.stale_after,
            head=args.head,
        )
        if args.json:
            _print_json(status)
        else:
            sys.stdout.write(json.dumps(status))
            sys.stdout.write("\n")
        return

    if args.command == "sync":
        imported = sync_submission_file(
            QueuePaths.from_queue_file(args.queue_file, queue_dir=args.queue_dir)
        )
        if args.json:
            _print_json({"imported": imported})
        else:
            sys.stdout.write(f"{imported}\n")
        return


if __name__ == "__main__":
    main()
