#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
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


def collect_status(queue_file: str, state_file, stale_after: int) -> dict:
    jobs = load_jobs(queue_file)
    status = {
        "queue_file": queue_file,
        "remaining": len(jobs),
        "jobs": jobs,
        "workers": [],
    }

    if state_file and os.path.exists(state_file):
        with open(state_file, "r") as handle:
            try:
                state = json.load(handle)
            except json.JSONDecodeError:
                state = {}
        workers = state.get("workers", {})
        if workers:
            now = time.time()
            for worker_id in sorted(workers):
                info = workers[worker_id]
                last_ping = info.get("last_ping")
                age = None if last_ping is None else int(now - last_ping)
                status_name = info.get("status", "unknown")
                if age is not None and stale_after > 0 and age > stale_after:
                    status_name = "stale"
                age_str = "unknown" if age is None else f"{age}s"
                job = info.get("job") or ""
                status["workers"].append(
                    {
                        "id": worker_id,
                        "status": status_name,
                        "age_str": age_str,
                        "job": job,
                    }
                )
    return status


def elide_middle(text: str, max_len: int):
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


def extract_job_label(job: str):
    patterns = [
        r"--save_outputs\s+(\S+)",
        r"\btee\s+(\S+)",
        r"-s\s+(\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, job)
        if match:
            return match.group(1).strip("\"'")

    matches = re.findall(
        r"([A-Za-z0-9_./-]+\.(?:jsonl|log|txt|json|yaml|yml|csv))", job
    )
    if matches:
        return matches[-1]

    return job.strip()


def job_display(job: str, width: int):
    label = extract_job_label(job)
    return elide_middle(label, max(12, width))


def fit_line(text: str, width: int):
    if width <= 0:
        return ""
    return elide_middle(text, width)


def build_display_rows(status: dict, head: int, width: int, now_str: str):
    rows = []
    rows.append(("Queue Status " + now_str, "header"))
    rows.append(("", "blank"))
    rows.append((f"Queue file: {status['queue_file']}", "meta"))
    rows.append((f"Remaining jobs: {status['remaining']}", "meta"))
    rows.append(("", "blank"))
    rows.append(("Next jobs:", "section"))

    if status["jobs"] and head > 0:
        for idx, job in enumerate(status["jobs"][:head], 1):
            label = job_display(job, max(10, width - 6))
            rows.append((f"{idx:>2}. {label}", "job"))
    else:
        rows.append(("  (none)", "dim"))

    rows.append(("", "blank"))
    rows.append(("Workers:", "section"))
    if status["workers"]:
        for worker in status["workers"]:
            base = f"{worker['id']} | {worker['status']} | last ping {worker['age_str']}"
            if worker["job"]:
                label = job_display(worker["job"], max(10, width - 10))
                base = f"{base} | {label}"
            style = f"worker_{worker['status']}"
            rows.append((base, style))
    else:
        rows.append(("  (none)", "dim"))

    return rows


def format_status_lines(status: dict, head: int, width: int, now_str: str):
    rows = build_display_rows(status, head, width, now_str)
    return [fit_line(text, width) for text, _style in rows]


def render_plain(status: dict, head: int) -> None:
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = format_status_lines(status, head, width, now_str)
    print("\n".join(lines))


def watch_plain(args: argparse.Namespace) -> None:
    while True:
        status = collect_status(args.file, args.state_file, args.stale_after)
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
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            curses.init_pair(5, curses.COLOR_RED, -1)

        while True:
            status = collect_status(args.file, args.state_file, args.stale_after)
            height, width = stdscr.getmaxyx()
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            rows = build_display_rows(status, args.head, width, now_str)
            hint = "Press q to quit"
            if height > 0:
                rows = rows[: max(0, height - 1)]
                if len(rows) < height:
                    rows.append((hint, "hint"))

            stdscr.bkgd(" ", curses.A_NORMAL)
            stdscr.erase()
            for idx, (line, style) in enumerate(rows[:height]):
                attr = curses.A_NORMAL
                if style == "header":
                    attr = curses.A_BOLD
                elif style == "section":
                    attr = curses.A_BOLD
                elif style == "dim":
                    attr = curses.A_DIM
                elif style == "worker_running":
                    attr = curses.A_BOLD
                    if has_colors:
                        attr = curses.color_pair(3) | curses.A_BOLD
                elif style == "worker_stale":
                    attr = curses.A_BOLD
                    if has_colors:
                        attr = curses.color_pair(5) | curses.A_BOLD
                elif style == "worker_idle":
                    attr = curses.A_BOLD
                    if has_colors:
                        attr = curses.color_pair(4) | curses.A_BOLD
                elif style == "hint":
                    attr = curses.A_DIM

                stdscr.addnstr(idx, 0, fit_line(line, max(0, width - 1)), max(0, width - 1), attr)
            stdscr.refresh()

            steps = max(1, int(args.interval / 0.1))
            for _ in range(steps):
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    return
                time.sleep(0.1)

    curses.wrapper(_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show queue status.")
    parser.add_argument("--file", required=True, help="Path to the queue file.")
    parser.add_argument("--head", type=int, default=5, help="Show next N jobs.")
    parser.add_argument("--state-file", help="Worker state JSON file.")
    parser.add_argument("--stale-after", type=int, default=0, help="Seconds before worker is stale.")
    parser.add_argument("--watch", action="store_true", help="Refresh display continuously.")
    parser.add_argument("--interval", type=float, default=10, help="Seconds between refreshes.")
    parser.add_argument("--curses", action="store_true", help="Use curses UI (implies --watch).")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses UI.")
    args = parser.parse_args()

    if args.curses:
        args.watch = True

    use_curses = False
    if args.watch and not args.no_curses and sys.stdout.isatty():
        term = os.environ.get("TERM", "")
        use_curses = term not in ("", "dumb", "unknown")

    if args.watch:
        if use_curses:
            try:
                watch_curses(args)
                return
            except Exception as exc:
                print(f"Warning: curses UI failed ({exc}); falling back to plain output.", file=sys.stderr)
        watch_plain(args)
        return

    status = collect_status(args.file, args.state_file, args.stale_after)
    render_plain(status, args.head)


if __name__ == "__main__":
    main()
