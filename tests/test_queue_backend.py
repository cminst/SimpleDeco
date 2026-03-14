from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from queue_backend import QueuePaths, claim_job, collect_status, complete_job, fail_job, reap_stale_jobs


def _mark_running_record_stale(queue_file: Path, heartbeat_age_seconds: int = 120) -> None:
    paths = QueuePaths.from_queue_file(str(queue_file))
    running_path = next(paths.running_dir.iterdir())
    payload = json.loads(running_path.read_text())
    payload["heartbeat_at"] = time.time() - heartbeat_age_seconds
    running_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_stale_claim_is_requeued_without_disappearing(tmp_path: Path) -> None:
    queue_file = tmp_path / "jobs.txt"
    queue_file.write_text("python3 -c 'print(1)'\n", encoding="utf-8")

    claimed = claim_job(str(queue_file), worker_id="worker-a")
    assert claimed is not None
    assert claimed["job"] == "python3 -c 'print(1)'"

    _mark_running_record_stale(queue_file)
    requeued = reap_stale_jobs(str(queue_file), stale_after=30, prepend=True)
    assert len(requeued) == 1
    assert requeued[0]["job_id"] == claimed["job_id"]

    status = collect_status(str(queue_file), stale_after=30, head=5)
    assert status["remaining"] == 1
    assert status["completed"] == 0
    assert status["failed"] == 0
    assert status["workers"] == []


def test_old_lease_cannot_complete_after_requeue(tmp_path: Path) -> None:
    queue_file = tmp_path / "jobs.txt"
    queue_file.write_text("python3 -c 'print(2)'\n", encoding="utf-8")

    first = claim_job(str(queue_file), worker_id="worker-a")
    assert first is not None

    _mark_running_record_stale(queue_file)
    reap_stale_jobs(str(queue_file), stale_after=30, prepend=True)

    second = claim_job(str(queue_file), worker_id="worker-b")
    assert second is not None
    assert second["job_id"] == first["job_id"]
    assert second["lease_id"] != first["lease_id"]

    assert not complete_job(
        str(queue_file),
        job_id=first["job_id"],
        lease_id=first["lease_id"],
        worker_id="worker-a",
    )

    running_status = collect_status(str(queue_file), stale_after=30, head=5)
    assert running_status["remaining"] == 0
    assert len(running_status["workers"]) == 1
    assert running_status["workers"][0]["worker_id"] == "worker-b"

    assert complete_job(
        str(queue_file),
        job_id=second["job_id"],
        lease_id=second["lease_id"],
        worker_id="worker-b",
    )

    final_status = collect_status(str(queue_file), stale_after=30, head=5)
    assert final_status["remaining"] == 0
    assert final_status["completed"] == 1
    assert final_status["failed"] == 0
    assert final_status["workers"] == []


def test_fail_retries_then_moves_to_failed(tmp_path: Path) -> None:
    queue_file = tmp_path / "jobs.txt"
    queue_file.write_text("python3 -c 'print(3)'\n", encoding="utf-8")

    first = claim_job(str(queue_file), worker_id="worker-a")
    assert first is not None
    result = fail_job(
        str(queue_file),
        job_id=first["job_id"],
        lease_id=first["lease_id"],
        worker_id="worker-a",
        max_retries=1,
        error="exit_code=1",
    )
    assert result["ok"] is True
    assert result["action"] == "retried"

    second = claim_job(str(queue_file), worker_id="worker-a")
    assert second is not None
    final = fail_job(
        str(queue_file),
        job_id=second["job_id"],
        lease_id=second["lease_id"],
        worker_id="worker-a",
        max_retries=1,
        error="exit_code=1",
    )
    assert final["ok"] is True
    assert final["action"] == "failed"

    status = collect_status(str(queue_file), stale_after=30, head=5)
    assert status["remaining"] == 0
    assert status["completed"] == 0
    assert status["failed"] == 1
