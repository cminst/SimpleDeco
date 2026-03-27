#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time

from queue_backend import collect_status, requeue_failed_jobs


def elide_middle(text: str, max_len: int) -> str:
    if max_len <= 3:
        return text[:max_len]
    if len(text) <= max_len:
        return text
    keep = max_len - 3
    keep_start = max(8, keep // 2)
    keep_end = keep - keep_start
    if keep_end < 4:
        keep_start = max(4, keep_start - (4 - keep_end))
        keep_end = keep - keep_start
    return f"{text[:keep_start]}...{text[-keep_end:]}"


def extract_job_label(job: str) -> str:
    patterns = [
        r"--output-file\s+(\S+)",
        r"\btee\s+(\S+)",
        r"-s\s+(\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, job)
        if match:
            return match.group(1).strip("\"'")

    matches = re.findall(r"([A-Za-z0-9_./-]+\.(?:jsonl|log|txt|json|yaml|yml|csv))", job)
    if matches:
        return matches[-1]

    return job.strip()


def job_display(job: str, width: int) -> str:
    return elide_middle(extract_job_label(job), max(12, width))


def fit_line(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return elide_middle(text, width)


def build_display_rows(status: dict, head: int, width: int, now_str: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.append((f"Queue Status {now_str}", "header"))
    rows.append(("", "blank"))
    rows.append((f"Queue file: {status['queue_file']}", "meta"))
    rows.append((f"Queue dir:  {status['queue_dir']}", "meta"))
    rows.append((f"Pending jobs:   {status['remaining']}", "meta"))
    rows.append((f"Completed jobs: {status['completed']}", "meta"))
    rows.append((f"Failed jobs:    {status['failed']}", "meta"))
    rows.append(("", "blank"))
    rows.append(("Next jobs:", "section"))

    if status["jobs"] and head > 0:
        for idx, job in enumerate(status["jobs"][:head], 1):
            label = job_display(job["job"], max(10, width - 6))
            retries = int(job.get("retries", 0))
            suffix = f" (retries={retries})" if retries else ""
            rows.append((f"{idx:>2}. {label}{suffix}", "job"))
    else:
        rows.append(("  (none)", "dim"))

    rows.append(("", "blank"))
    rows.append(("Running leases:", "section"))
    workers = status.get("workers", [])
    if workers:
        for worker in workers:
            label = job_display(worker["job"], max(10, width - 24))
            state = "stale" if worker.get("stale") else "running"
            base = f"{worker.get('worker_id') or 'unknown'} | {state} | age {worker.get('age_seconds', 0)}s | {label}"
            if worker.get("progress"):
                base = f"{base} | {worker['progress']}"
            rows.append((base, f"worker_{state}"))
    else:
        rows.append(("  (none)", "dim"))

    return rows


def format_status_lines(status: dict, head: int, width: int, now_str: str) -> list[str]:
    rows = build_display_rows(status, head, width, now_str)
    return [fit_line(text, width) for text, _style in rows]


def render_plain(status: dict, head: int) -> None:
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = format_status_lines(status, head, width, now_str)
    print("\n".join(lines))


def watch_plain(args: argparse.Namespace) -> None:
    while True:
        status = collect_status(args.file, stale_after=args.stale_after, head=args.head)
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        render_plain(status, args.head)
        time.sleep(args.interval)


def watch_curses(args: argparse.Namespace) -> None:
    import curses

    def _run(stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        has_colors = False
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except Exception:
                pass
            has_colors = True
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(5, curses.COLOR_RED, -1)
        status_message = ""
        status_message_until = 0.0

        while True:
            status = collect_status(args.file, stale_after=args.stale_after, head=args.head)
            height, width = stdscr.getmaxyx()
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            rows = build_display_rows(status, args.head, width, now_str)
            if time.time() < status_message_until and status_message:
                hint = status_message
            else:
                hint = "Press r to requeue failed jobs, q to quit"
            if height > 0:
                rows = rows[: max(0, height - 1)]
                if len(rows) < height:
                    rows.append((hint, "hint"))

            stdscr.bkgd(" ", curses.A_NORMAL)
            stdscr.erase()
            for idx, (line, style) in enumerate(rows[:height]):
                attr = curses.A_NORMAL
                if style in {"header", "section"}:
                    attr = curses.A_BOLD
                elif style == "dim":
                    attr = curses.A_DIM
                elif style == "worker_running":
                    attr = curses.color_pair(3) | curses.A_BOLD if has_colors else curses.A_BOLD
                elif style == "worker_stale":
                    attr = curses.color_pair(5) | curses.A_BOLD if has_colors else curses.A_BOLD
                elif style == "hint":
                    attr = curses.A_DIM
                stdscr.addnstr(idx, 0, fit_line(line, max(0, width - 1)), max(0, width - 1), attr)
            stdscr.refresh()

            steps = max(1, int(args.interval / 0.1))
            for _ in range(steps):
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    return
                if key in (ord("r"), ord("R")):
                    try:
                        requeued = requeue_failed_jobs(args.file)
                        count = len(requeued)
                        if count == 0:
                            status_message = "No failed jobs to requeue"
                        elif count == 1:
                            status_message = "Requeued 1 failed job"
                        else:
                            status_message = f"Requeued {count} failed jobs"
                    except Exception as exc:
                        status_message = f"Requeue failed: {exc}"
                    status_message_until = time.time() + max(2.0, min(args.interval, 5.0))
                    break
                time.sleep(0.1)

    curses.wrapper(_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show durable queue status.")
    parser.add_argument("--file", required=True, help="Queue inbox file.")
    parser.add_argument("--head", type=int, default=5, help="Show next N pending jobs.")
    parser.add_argument("--state-file", help="Ignored legacy flag.")
    parser.add_argument("--completed-file", help="Ignored legacy flag.")
    parser.add_argument("--failed-file", help="Ignored legacy flag.")
    parser.add_argument("--stale-after", type=int, default=0, help="Seconds before a running lease is shown as stale.")
    parser.add_argument("--watch", action="store_true", help="Refresh display continuously.")
    parser.add_argument("--interval", type=float, default=10, help="Seconds between refreshes.")
    parser.add_argument("--curses", action="store_true", help="Use curses UI (implies --watch).")
    parser.add_argument("--no-curses", action="store_true", help="Force plain-text watch mode.")
    args = parser.parse_args()

    if args.curses:
        args.watch = True

    use_curses = False
    if args.watch and not args.no_curses and sys.stdout.isatty():
        term = os.environ.get("TERM", "")
        if term and term.lower() != "dumb":
            use_curses = args.curses or True

    if args.watch:
        if use_curses:
            watch_curses(args)
        else:
            watch_plain(args)
        return

    status = collect_status(args.file, stale_after=args.stale_after, head=args.head)
    render_plain(status, args.head)


if __name__ == "__main__":
    main()
