#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_str(name, default):
    return os.environ.get(name, default)


def _build_cmd(parts):
    return " ".join(shlex.quote(part) for part in parts)


PROGRESS_RE = re.compile(r"Processed prompts:\s*(\d+)%.*?(\d+)\s*/\s*(\d+)")
SPEED_RE = re.compile(r"input:\s*([0-9.]+)\s*toks/s,\s*output:\s*([0-9.]+)\s*toks/s")


class QueueWorker:
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.queue_host = args.queue_host
        self.queue_file = args.queue_file
        self.queue_pop_script = args.queue_pop_script
        self.queue_append_script = args.queue_append_script
        self.queue_worker_state_script = args.queue_worker_state_script
        self.worker_state_file = args.worker_state_file
        self.failed_file = args.failed_file
        self.ssh_opts = args.ssh_opts
        self.ssh_pass = args.ssh_pass
        self.ssh_accept_new = args.ssh_accept_new
        self.gpu_id = args.gpu_id
        self.sleep_sec = args.sleep_sec
        self.exit_on_empty = args.exit_on_empty
        self.stop_on_fail = args.stop_on_fail
        self.requeue_on_fail = args.requeue_on_fail
        self.max_retries = args.max_retries
        self.max_jobs_to_run = args.max_jobs_to_run
        self.worker_id = args.worker_id
        self.ping_interval = args.ping_interval
        self.stale_after = args.stale_after
        self.clear_on_exit = args.clear_on_exit
        self.progress_ping_interval = args.progress_ping_interval
        self.prune_after = args.prune_after

        self._current_job = ""
        self._job_started = None
        self._job_running = False
        self._progress = ""
        self._last_progress_ping = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._run_count = 0

    def _ssh_base(self):
        cmd = []
        if self.ssh_pass:
            cmd.extend(["sshpass", "-p", self.ssh_pass])
        cmd.append("ssh")
        ssh_opts = self.ssh_opts
        if self.ssh_accept_new:
            ssh_opts = f"{ssh_opts} -o StrictHostKeyChecking=accept-new"
        if ssh_opts:
            cmd.extend(shlex.split(ssh_opts))
        cmd.append(self.queue_host)
        return cmd

    def _run_remote(self, cmd_parts, input_text=None, capture=False, check=True):
        cmd = self._ssh_base() + [_build_cmd(cmd_parts)]
        result = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=capture,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result

    def _run_local(self, cmd_parts, input_text=None, capture=False, check=True):
        result = subprocess.run(
            cmd_parts,
            input=input_text,
            text=True,
            capture_output=capture,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd_parts, result.stdout, result.stderr)
        return result

    def _run_cmd(self, cmd_parts, input_text=None, capture=False, check=True):
        if self.queue_host:
            return self._run_remote(cmd_parts, input_text=input_text, capture=capture, check=check)
        return self._run_local(cmd_parts, input_text=input_text, capture=capture, check=check)

    def _pop_job(self):
        cmd = [
            "python3",
            self.queue_pop_script,
            "--file",
            self.queue_file,
        ]
        result = self._run_cmd(cmd, capture=True, check=True)
        return (result.stdout or "").strip("\r\n")

    def _append_job(self, job_line, prepend=False, failed=False):
        target_file = self.failed_file if failed else self.queue_file
        cmd = [
            "python3",
            self.queue_append_script,
            "--file",
            target_file,
        ]
        if prepend:
            cmd.append("--prepend")
        cmd.append("--stdin")
        self._run_cmd(cmd, input_text=job_line, check=True)

    def _send_ping(self, status, job_line=None, progress=None):
        if not self.queue_worker_state_script:
            return
        cmd = [
            "python3",
            self.queue_worker_state_script,
            "--state-file",
            self.worker_state_file,
            "--worker-id",
            self.worker_id,
            "--status",
            status,
            "--hostname",
            socket.gethostname(),
            "--pid",
            str(os.getpid()),
            "--reap-stale",
            "--stale-after",
            str(self.stale_after),
            "--prune-after",
            str(self.prune_after),
            "--queue-file",
            self.queue_file,
            "--queue-append-script",
            self.queue_append_script,
            "--prepend-stale",
        ]
        input_text = None
        if job_line:
            cmd.append("--job-stdin")
            input_text = job_line
        if progress:
            cmd.extend(["--progress", progress])
        try:
            self._run_cmd(cmd, input_text=input_text, capture=False, check=True)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"Worker ping failed: {exc}\n")

    def _ping_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                status = "running" if self._job_running else "idle"
                job_line = self._current_job if self._job_running else None
                progress = self._progress if self._job_running else None
            self._send_ping(status, job_line=job_line, progress=progress)
            self._stop_event.wait(self.ping_interval)

    @staticmethod
    def _parse_retry(job_line):
        match = re.search(r"#\s*retry=(\d+)", job_line)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _strip_retry(job_line):
        if "# retry=" in job_line:
            job_line = job_line.split("# retry=")[0]
        return job_line.rstrip()

    @staticmethod
    def _extract_progress(line):
        match = PROGRESS_RE.search(line)
        if not match:
            return None
        percent, done, total = match.groups()
        speed = SPEED_RE.search(line)
        if speed:
            return f"{percent}% {done}/{total} in {speed.group(1)} out {speed.group(2)} toks/s"
        return f"{percent}% {done}/{total}"

    def _maybe_update_progress(self, line):
        progress = self._extract_progress(line)
        if not progress:
            return
        now = time.time()
        should_ping = False
        with self._lock:
            if progress == self._progress:
                return
            self._progress = progress
            if (
                self.progress_ping_interval > 0
                and now - self._last_progress_ping >= self.progress_ping_interval
                and self._job_running
            ):
                self._last_progress_ping = now
                should_ping = True
                job_line = self._current_job
            else:
                job_line = None
        if should_ping:
            self._send_ping("running", job_line=job_line, progress=progress)

    def _stream_output(self, proc):
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

    def _run_job(self, job_line):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        command = f'cd "{self.root_dir}" && set -o pipefail && {job_line}'

        with self._lock:
            self._current_job = job_line
            self._job_running = True
            self._job_started = time.time()
            self._progress = ""

        try:
            self._send_ping("running", job_line=job_line)
        except KeyboardInterrupt:
            pass

        proc = subprocess.Popen(
            ["bash", "-lc", command],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
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

        with self._lock:
            self._current_job = ""
            self._job_running = False
            self._job_started = None
            self._progress = ""
        try:
            self._send_ping("idle")
        except KeyboardInterrupt:
            pass
        return returncode, interrupted

    def _clear_worker(self):
        cmd = [
            "python3",
            self.queue_worker_state_script,
            "--state-file",
            self.worker_state_file,
            "--worker-id",
            self.worker_id,
            "--clear",
        ]
        try:
            self._run_cmd(cmd, capture=False, check=True)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(f"Worker clear failed: {exc}\n")

    def run(self):
        self._send_ping("idle")
        ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        ping_thread.start()

        try:
            while True:
                job = self._pop_job()
                job = job.strip()
                if not job:
                    if self.exit_on_empty:
                        print("Queue empty. Exiting.")
                        break
                    time.sleep(self.sleep_sec)
                    continue

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running: {job}")
                returncode, interrupted = self._run_job(job)

                if interrupted:
                    print(f"Interrupted. Re-queuing: {job}", file=sys.stderr)
                    self._append_job(job, prepend=True)
                    break

                if returncode != 0:
                    print(f"Job failed: {job}", file=sys.stderr)
                    if self.requeue_on_fail:
                        retries = self._parse_retry(job)
                        if retries < self.max_retries:
                            base_job = self._strip_retry(job)
                            requeued = f"{base_job} # retry={retries + 1}"
                            print(
                                f"Re-queuing (attempt {retries + 1}/{self.max_retries}): {base_job}"
                            )
                            self._append_job(requeued)
                        else:
                            print("Max retries reached. Recording failure.")
                            self._append_job(job, failed=True)
                    else:
                        print("Re-queue disabled; recording failure.")
                        self._append_job(job, failed=True)
                    print("Exiting after failure.")
                    break

                self._run_count += 1
                if self.max_jobs_to_run > 0 and self._run_count >= self.max_jobs_to_run:
                    print(f"Reached MAX_JOBS_TO_RUN={self.max_jobs_to_run}. Exiting.")
                    break
        finally:
            self._stop_event.set()
            if self.clear_on_exit:
                try:
                    self._clear_worker()
                except KeyboardInterrupt:
                    pass


def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Queue worker with heartbeat tracking.")
    parser.add_argument("--root-dir", default=root_dir, help="Repository root.")
    parser.add_argument("--queue-host", default=_env_str("QUEUE_HOST", ""), help="Queue host.")
    parser.add_argument("--queue-file", default=_env_str("QUEUE_FILE", ""))
    parser.add_argument("--queue-pop-script", default=_env_str("QUEUE_POP_SCRIPT", ""))
    parser.add_argument("--queue-append-script", default=_env_str("QUEUE_APPEND_SCRIPT", ""))
    parser.add_argument(
        "--queue-worker-state-script",
        default=_env_str("QUEUE_WORKER_STATE_SCRIPT", ""),
    )
    parser.add_argument(
        "--worker-state-file",
        default=_env_str("WORKER_STATE_FILE", ""),
    )
    parser.add_argument("--failed-file", default=_env_str("FAILED_FILE", ""))
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
    parser.add_argument("--gpu-id", type=int, default=_env_int("GPU_ID", 0))
    parser.add_argument("--sleep-sec", type=int, default=_env_int("SLEEP_SEC", 30))
    parser.add_argument("--exit-on-empty", type=int, default=_env_int("EXIT_ON_EMPTY", 1))
    parser.add_argument("--stop-on-fail", type=int, default=_env_int("STOP_ON_FAIL", 0))
    parser.add_argument("--requeue-on-fail", type=int, default=_env_int("REQUEUE_ON_FAIL", 1))
    parser.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 3))
    parser.add_argument("--max-jobs-to-run", type=int, default=_env_int("MAX_JOBS_TO_RUN", 0))
    parser.add_argument(
        "--worker-id",
        default=_env_str("WORKER_ID", f"{socket.gethostname()}-{os.getpid()}"),
    )
    parser.add_argument("--ping-interval", type=int, default=_env_int("PING_INTERVAL", 600))
    parser.add_argument("--stale-after", type=int, default=_env_int("STALE_AFTER", 0))
    parser.add_argument(
        "--progress-ping-interval",
        type=int,
        default=_env_int("PROGRESS_PING_INTERVAL", 60),
    )
    parser.add_argument(
        "--prune-after",
        type=int,
        default=_env_int("PRUNE_AFTER", 7200),
    )
    parser.add_argument("--clear-on-exit", type=int, default=_env_int("CLEAR_ON_EXIT", 1))
    args = parser.parse_args()

    if args.queue_host:
        remote_root = args.queue_remote_root
        if not args.queue_pop_script:
            args.queue_pop_script = f"{remote_root}/script/queue_pop_job.py"
        if not args.queue_append_script:
            args.queue_append_script = f"{remote_root}/script/queue_append_job.py"
        if not args.queue_worker_state_script:
            args.queue_worker_state_script = f"{remote_root}/script/queue_worker_state.py"
        if not args.worker_state_file:
            args.worker_state_file = f"{remote_root}/jobs/worker_state.json"
        if not args.failed_file:
            args.failed_file = f"{remote_root}/jobs/failed_jobs.txt"
    else:
        if not args.queue_pop_script:
            args.queue_pop_script = f"{root_dir}/script/queue_pop_job.py"
        if not args.queue_append_script:
            args.queue_append_script = f"{root_dir}/script/queue_append_job.py"
        if not args.queue_worker_state_script:
            args.queue_worker_state_script = f"{root_dir}/script/queue_worker_state.py"
        if not args.worker_state_file:
            args.worker_state_file = f"{root_dir}/jobs/worker_state.json"
        if not args.failed_file:
            args.failed_file = f"{root_dir}/jobs/failed_jobs.txt"

    if not args.queue_file:
        parser.error("--queue-file is required (or set QUEUE_FILE).")

    if not args.stale_after:
        args.stale_after = max(2 * args.ping_interval, args.ping_interval + 60)

    worker = QueueWorker(args)
    worker.run()


if __name__ == "__main__":
    main()
