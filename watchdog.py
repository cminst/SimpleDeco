import argparse
import subprocess
import time
from pathlib import Path


def get_jsonl_file_set(path: str) -> set[str]:
    root = Path(path)
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.jsonl")
        if p.is_file()
    }


def run_upload(directory_to_watch: str, repo_id: str, includes: list[str] | None = None):
    cmd = ["hf", "upload", repo_id, directory_to_watch, "--repo-type", "dataset"]
    if includes:
        for rel_path in includes:
            cmd.extend(["--include", rel_path])
    else:
        cmd.extend(["--include", "*.jsonl"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() or str(e)


def run_download(repo_id: str, dest_dir: str):
    cmd = ["hf", "download", repo_id, "--repo-type", "dataset", "--local-dir", dest_dir, "*.jsonl"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() or str(e)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a directory for new .jsonl files and upload them to Hugging Face"
    )
    parser.add_argument(
        "--dir",
        default="ckpt_gptoss20b",
        help="Directory to monitor for .jsonl files",
    )
    parser.add_argument(
        "--repo",
        default="cminst/SimpleDeco-gptoss20b",
        help="Destination Hugging Face repo id",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Also download from HF repo every minute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval_str = str(int(args.check_interval)) if args.check_interval.is_integer() else str(args.check_interval)
    path = Path(args.dir)

    if not path.exists():
        print(f"Error: {args.dir} does not exist.")
        return

    log("Initial upload...")
    success, out = run_upload(args.dir, args.repo)
    if success:
        log(f"Done. {out}")
    else:
        log(f"Failed: {out}")

    known_files = get_jsonl_file_set(args.dir)
    log(
        "Tracking "
        f"{len(known_files)} existing file(s). Polling every "
        f"{interval_str}s."
    )

    sync_interval = 60
    last_sync_time = time.time()

    while True:
        time.sleep(args.check_interval)
        current_files = get_jsonl_file_set(args.dir)
        new_files = sorted(current_files - known_files)

        now = time.time()
        should_sync = args.sync and (now - last_sync_time) >= sync_interval

        if not new_files and not should_sync:
            continue

        if should_sync:
            log("Syncing from HF...")
            success, out = run_download(args.repo, args.dir)
            if success:
                log(f"Sync done. {out}")
            else:
                log(f"Sync failed: {out}")
            known_files = get_jsonl_file_set(args.dir)
            last_sync_time = time.time()

        if not new_files:
            continue

        log(
            f"{len(new_files)} new file(s): {', '.join(new_files[:5])}"
            + (f" ... and {len(new_files)-5} more" if len(new_files) > 5 else "")
        )

        success, out = run_upload(args.dir, args.repo, new_files)
        if success:
            known_files = current_files
            log(f"Uploaded. {out}")
        else:
            log(f"Upload failed: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping.")
