#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

from queue_backend import claim_job, complete_job, fail_job, heartbeat_job, reap_stale_jobs, release_job


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


PROGRESS_RE = re.compile(r"Processed prompts:\s*(\d+)%.*?(\d+)\s*/\s*(\d+)")
SPEED_RE = re.compile(r"input:\s*([0-9.]+)\s*toks/s,\s*output:\s*([0-9.]+)\s*toks/s")


class QueueWorker:
    def __init__(self, args: argparse.Namespace):
        self.root_dir = args.root_dir
        self.queue_host = args.queue_host
        self.queue_file = args.queue_file
        self.queue_backend_script = args.queue_backend_script
        self.ssh_opts = args.ssh_opts
        self.ssh_pass = args.ssh_pass
        self.ssh_accept_new = args.ssh_accept_new
        self.gpu_id = args.gpu_id
        self.sleep_sec = args.sleep_sec
        self.exit_on_empty = bool(args.exit_on_empty)
        self.stop_on_fail = bool(args.stop_on_fail)
        self.requeue_on_fail = bool(args.requeue_on_fail)
        self.max_retries = args.max_retries
        self.max_jobs_to_run = args.max_jobs_to_run
        self.worker_id = args.worker_id
        self.ping_interval = args.ping_interval
        self.stale_after = args.stale_after
        self.progress_ping_interval = args.progress_ping_interval
        self.hostname = socket.gethostname()

        self._current_job: dict[str, object] | None = None
        self._lease_lost = False
        self._last_progress_ping = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._run_count = 0

    def _ssh_base(self) -> list[str]:
        cmd: list[str] = []
        if self.ssh_pass:
            cmd.extend(["sshpass", "-p", self.ssh_pass])
        cmd.append("ssh")
        ssh_opts = self.ssh_opts
        if self.ssh_accept_new:
            ssh_opts = f"{ssh_opts} -o StrictHostKeyChecking=accept-new".strip()
        if ssh_opts:
            cmd.extend(shlex.split(ssh_opts))
        cmd.append(self.queue_host)
        return cmd

    def _run_remote(self, cmd_parts: list[str], capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = self._ssh_base() + [_build_cmd(cmd_parts)]
        result = subprocess.run(cmd, text=True, capture_output=capture)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result

    def _queue_json_remote(self, subcommand: list[str]) -> object | None:
        cmd = ["python3", self.queue_backend_script, "--queue-file", self.queue_file, *subcommand, "--json"]
        result = self._run_remote(cmd, capture=True, check=True)
        stdout = (result.stdout or "").strip()
        return json.loads(stdout) if stdout else None

    def _queue_remote(self, subcommand: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["python3", self.queue_backend_script, "--queue-file", self.queue_file, *subcommand]
        return self._run_remote(cmd, capture=True, check=check)

    def _reap_stale(self) -> None:
        if self.stale_after <= 0:
            return
        try:
            if self.queue_host:
                self._queue_remote(["reap", "--stale-after", str(self.stale_after), "--prepend"], check=True)
            else:
                reap_stale_jobs(self.queue_file, stale_after=self.stale_after, prepend=True)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"Stale reap failed: {exc}\n")

    def _claim_job(self) -> dict[str, object] | None:
        self._reap_stale()
        if self.queue_host:
            payload = self._queue_json_remote(
                ["claim", "--worker-id", self.worker_id, "--hostname", self.hostname]
            )
            return payload if isinstance(payload, dict) else None
        return claim_job(self.queue_file, worker_id=self.worker_id, hostname=self.hostname)

    def _heartbeat(self, progress: str | None = None) -> bool:
        with self._lock:
            current = dict(self._current_job or {})
        if not current:
            self._reap_stale()
            return True
        job_id = str(current["job_id"])
        lease_id = str(current["lease_id"])
        try:
            if self.queue_host:
                cmd = [
                    "heartbeat",
                    "--job-id",
                    job_id,
                    "--lease-id",
                    lease_id,
                    "--worker-id",
                    self.worker_id,
                    "--hostname",
                    self.hostname,
                ]
                if progress is not None:
                    cmd.extend(["--progress", progress])
                self._queue_remote(cmd, check=True)
                ok = True
            else:
                ok = heartbeat_job(
                    self.queue_file,
                    job_id=job_id,
                    lease_id=lease_id,
                    worker_id=self.worker_id,
                    hostname=self.hostname,
                    progress=progress,
                )
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"Heartbeat failed: {exc}\n")
            ok = False
        if not ok:
            with self._lock:
                self._lease_lost = True
        return ok

    def _complete_job(self, job_id: str, lease_id: str) -> bool:
        if self.queue_host:
            try:
                self._queue_remote(
                    [
                        "complete",
                        "--job-id",
                        job_id,
                        "--lease-id",
                        lease_id,
                        "--worker-id",
                        self.worker_id,
                    ],
                    check=True,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        return complete_job(
            self.queue_file,
            job_id=job_id,
            lease_id=lease_id,
            worker_id=self.worker_id,
        )

    def _release_job(self, job_id: str, lease_id: str, prepend: bool) -> bool:
        if self.queue_host:
            try:
                cmd = [
                    "release",
                    "--job-id",
                    job_id,
                    "--lease-id",
                    lease_id,
                    "--worker-id",
                    self.worker_id,
                ]
                if prepend:
                    cmd.append("--prepend")
                self._queue_remote(cmd, check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        return release_job(
            self.queue_file,
            job_id=job_id,
            lease_id=lease_id,
            worker_id=self.worker_id,
            prepend=prepend,
        )

    def _fail_job(self, job_id: str, lease_id: str, error: str) -> dict[str, object]:
        retries = self.max_retries if self.requeue_on_fail else -1
        if self.queue_host:
            payload = self._queue_json_remote(
                [
                    "fail",
                    "--job-id",
                    job_id,
                    "--lease-id",
                    lease_id,
                    "--worker-id",
                    self.worker_id,
                    "--max-retries",
                    str(retries),
                    "--error",
                    error,
                ]
            )
            return payload if isinstance(payload, dict) else {"ok": False, "action": "missing"}
        return fail_job(
            self.queue_file,
            job_id=job_id,
            lease_id=lease_id,
            worker_id=self.worker_id,
            max_retries=retries,
            error=error,
        )

    @staticmethod
    def _extract_progress(line: str) -> str | None:
        match = PROGRESS_RE.search(line)
        if not match:
            return None
        percent, done, total = match.groups()
        speed = SPEED_RE.search(line)
        if speed:
            return f"{percent}% {done}/{total} in {speed.group(1)} out {speed.group(2)} toks/s"
        return f"{percent}% {done}/{total}"

    def _maybe_update_progress(self, line: str) -> None:
        progress = self._extract_progress(line)
        if not progress:
            return
        should_ping = False
        with self._lock:
            if not self._current_job:
                return
            if self._current_job.get("progress") == progress:
                return
            self._current_job["progress"] = progress
            now = time.time()
            if (
                self.progress_ping_interval > 0
                and now - self._last_progress_ping >= self.progress_ping_interval
            ):
                self._last_progress_ping = now
                should_ping = True
        if should_ping:
            self._heartbeat(progress=progress)

    def _stream_output(self, proc: subprocess.Popen[bytes]) -> None:
        buffer = b""
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            buffer += chunk
            while True:
                idx_r = buffer.find(b"\r")
                idx_n = buffer.find(b"\n")
                if idx_r == -1 and idx_n == -1:
                    break
                if idx_r == -1:
                    idx = idx_n
                elif idx_n == -1:
                    idx = idx_r
                else:
                    idx = min(idx_r, idx_n)
                line = buffer[:idx]
                buffer = buffer[idx + 1 :]
                if line:
                    self._maybe_update_progress(line.decode(errors="ignore"))
        if buffer:
            self._maybe_update_progress(buffer.decode(errors="ignore"))

    def _ping_loop(self) -> None:
        while not self._stop_event.is_set():
            self._heartbeat()
            self._stop_event.wait(self.ping_interval)

    def _run_job(self, record: dict[str, object]) -> tuple[int, bool]:
        job_line = str(record["job"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        command = f'cd "{self.root_dir}" && set -o pipefail && {job_line}'

        with self._lock:
            self._current_job = {
                "job": job_line,
                "job_id": str(record["job_id"]),
                "lease_id": str(record["lease_id"]),
                "progress": "",
            }
            self._lease_lost = False
            self._last_progress_ping = 0.0

        proc = subprocess.Popen(
            ["bash", "-lc", command],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        stream_thread = threading.Thread(target=self._stream_output, args=(proc,), daemon=True)
        stream_thread.start()

        interrupted = False
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            proc.send_signal(signal.SIGINT)
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = 130

        stream_thread.join(timeout=5)
        return returncode, interrupted

    def run(self) -> None:
        self._reap_stale()
        ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        ping_thread.start()

        try:
            while True:
                record = self._claim_job()
                if not record:
                    if self.exit_on_empty:
                        print("Queue empty. Exiting.", flush=True)
                        break
                    time.sleep(self.sleep_sec)
                    continue

                job_line = str(record["job"])
                job_id = str(record["job_id"])
                lease_id = str(record["lease_id"])
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running: {job_line}", flush=True)
                returncode, interrupted = self._run_job(record)

                with self._lock:
                    lease_lost = self._lease_lost
                    self._current_job = None

                if interrupted:
                    print(f"Interrupted. Releasing job back to the front: {job_line}", file=sys.stderr, flush=True)
                    self._release_job(job_id, lease_id, prepend=True)
                    break

                if lease_lost:
                    print(
                        f"Lease was lost while running; leaving queue state untouched for {job_line}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                if returncode != 0:
                    result = self._fail_job(job_id, lease_id, error=f"exit_code={returncode}")
                    action = str(result.get("action", "missing"))
                    if action == "retried":
                        retries = result.get("record", {}).get("retries") if isinstance(result.get("record"), dict) else None
                        print(f"Job failed and was requeued (retries={retries}): {job_line}", file=sys.stderr, flush=True)
                    elif action == "failed":
                        print(f"Job failed permanently: {job_line}", file=sys.stderr, flush=True)
                    else:
                        print(f"Job failed but queue lease was already gone: {job_line}", file=sys.stderr, flush=True)
                    if self.stop_on_fail:
                        break
                    continue

                if not self._complete_job(job_id, lease_id):
                    print(
                        f"Completed job could not be acknowledged because the lease no longer exists: {job_line}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

                self._run_count += 1
                if self.max_jobs_to_run > 0 and self._run_count >= self.max_jobs_to_run:
                    print(f"Reached MAX_JOBS_TO_RUN={self.max_jobs_to_run}. Exiting.", flush=True)
                    break
        finally:
            self._stop_event.set()


def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Durable queue worker with lease-based claims.")
    parser.add_argument("--root-dir", default=root_dir, help="Repository root.")
    parser.add_argument("--queue-host", default=_env_str("QUEUE_HOST", ""), help="Queue host.")
    parser.add_argument("--queue-file", default=_env_str("QUEUE_FILE", ""))
    parser.add_argument("--queue-backend-script", default=_env_str("QUEUE_BACKEND_SCRIPT", ""))
    parser.add_argument(
        "--queue-remote-root",
        default=_env_str("QUEUE_REMOTE_ROOT", "SimpleDeco"),
        help="Remote repo root when using --queue-host.",
    )
    parser.add_argument("--ssh-opts", default=_env_str("SSH_OPTS", ""))
    parser.add_argument("--ssh-pass", default=_env_str("SSH_PASS", ""))
    parser.add_argument(
        "--ssh-accept-new",
        action="store_true",
        default=_env_int("SSH_ACCEPT_NEW", 0) == 1,
        help="Auto-accept new SSH host keys.",
    )
    parser.add_argument("--gpu-id", type=int, default=_env_int("GPU_ID", 0) or 0)
    parser.add_argument("--sleep-sec", type=int, default=_env_int("SLEEP_SEC", 15) or 15)
    parser.add_argument("--exit-on-empty", type=int, default=_env_int("EXIT_ON_EMPTY", None))
    parser.add_argument("--stop-on-fail", type=int, default=_env_int("STOP_ON_FAIL", 0) or 0)
    parser.add_argument("--requeue-on-fail", type=int, default=_env_int("REQUEUE_ON_FAIL", 1) or 1)
    parser.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 3) or 3)
    parser.add_argument("--max-jobs-to-run", type=int, default=_env_int("MAX_JOBS_TO_RUN", 0) or 0)
    parser.add_argument(
        "--worker-id",
        default=_env_str("WORKER_ID", f"{socket.gethostname()}-{os.getpid()}"),
    )
    parser.add_argument("--ping-interval", type=int, default=_env_int("PING_INTERVAL", 60) or 60)
    parser.add_argument("--stale-after", type=int, default=_env_int("STALE_AFTER", 0) or 0)
    parser.add_argument(
        "--progress-ping-interval",
        type=int,
        default=_env_int("PROGRESS_PING_INTERVAL", 60) or 60,
    )
    args = parser.parse_args()

    if args.queue_host:
        remote_root = args.queue_remote_root
        if not args.queue_backend_script:
            args.queue_backend_script = f"{remote_root}/script/queue_backend.py"
    else:
        if not args.queue_backend_script:
            args.queue_backend_script = f"{root_dir}/script/queue_backend.py"

    if not args.queue_file:
        parser.error("--queue-file is required (or set QUEUE_FILE).")

    if args.exit_on_empty is None:
        args.exit_on_empty = 0 if not args.queue_host else 1

    if not args.stale_after:
        args.stale_after = max(3 * args.ping_interval, args.ping_interval + 60)

    worker = QueueWorker(args)
    worker.run()


if __name__ == "__main__":
    main()
