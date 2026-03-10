#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import textwrap
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


def wrap_with_prefix(text: str, prefix: str, width: int):
    if width <= len(prefix) + 2:
        return [f"{prefix}{text[: max(0, width - len(prefix))]}"]
    wrapped = textwrap.wrap(text, width=width - len(prefix)) or [""]
    lines = [prefix + wrapped[0]]
    indent = " " * len(prefix)
    for chunk in wrapped[1:]:
        lines.append(indent + chunk)
    return lines


def format_status_lines(status: dict, head: int, width: int, now_str: str):
    lines = []
    lines.append(now_str)
    lines.append("")
    lines.append(f"Queue file: {status['queue_file']}")
    lines.append(f"Remaining jobs: {status['remaining']}")

    if head > 0:
        if status["jobs"]:
            lines.append("Next jobs:")
            for job in status["jobs"][:head]:
                lines.extend(wrap_with_prefix(job, "- ", width))
        else:
            lines.append("Next jobs: (none)")

    if status["workers"]:
        lines.append("")
        lines.append("Workers:")
        for worker in status["workers"]:
            base = f"{worker['id']} | {worker['status']} | last ping {worker['age_str']}"
            lines.extend(wrap_with_prefix(base, "- ", width))
            if worker["job"]:
                lines.extend(wrap_with_prefix(worker["job"], "  job: ", width))
    else:
        lines.append("")
        lines.append("Workers: (none)")

    return lines


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

        while True:
            status = collect_status(args.file, args.state_file, args.stale_after)
            height, width = stdscr.getmaxyx()
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = format_status_lines(status, args.head, width, now_str)
            hint = "Press q to quit"
            if height > 0:
                lines = lines[: max(0, height - 1)]
                if len(lines) < height:
                    lines.append(hint)

            stdscr.erase()
            for idx, line in enumerate(lines[:height]):
                stdscr.addnstr(idx, 0, line, max(0, width - 1))
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
