import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_REPO_TYPE = "dataset"
JSONL_GLOB = "**/*.jsonl"


def get_jsonl_file_set(path: str, focus_path: Path | None = None) -> set[str]:
    root = Path(path).resolve()
    search_path = (focus_path or root).resolve()

    if not search_path.exists():
        return set()

    return {
        p.relative_to(root).as_posix()
        for p in search_path.rglob("*.jsonl")
        if p.is_file()
    }


def _focus_pattern(focus_relative: str | None) -> list[str] | None:
    if not focus_relative or focus_relative == ".":
        return None

    return [f"{focus_relative}/{JSONL_GLOB}"]


def _resolve_focus_dir(directory_to_watch: Path, focus: str | None) -> tuple[Path, str | None]:
    if not focus:
        return directory_to_watch, None

    focused = Path(focus)
    if not focused.is_absolute():
        focused = directory_to_watch / focused

    focused = focused.resolve()

    if not focused.exists():
        raise ValueError(f"Focus directory '{focus}' does not exist.")
    if not focused.is_dir():
        raise ValueError(f"Focus path '{focus}' is not a directory.")

    try:
        focus_relative = focused.relative_to(directory_to_watch).as_posix()
    except ValueError as exc:
        raise ValueError(f"Focus directory '{focus}' must be inside --dir '{directory_to_watch}'.") from exc

    return focused, focus_relative


def run_upload(directory_to_watch: str, repo_id: str, includes: list[str] | None = None):
    api = HfApi()
    allow_patterns = includes if includes else JSONL_GLOB

    try:
        commit_info = api.upload_folder(
            repo_id=repo_id,
            folder_path=directory_to_watch,
            repo_type=DATASET_REPO_TYPE,
            allow_patterns=allow_patterns,
        )
        return True, str(commit_info.repo_url)
    except Exception as e:
        return False, str(e)


def run_download(repo_id: str, dest_dir: str, includes: list[str] | None = None):
    print(f"Running snapshot_download for {repo_id} -> {Path(dest_dir).resolve()}")
    allow_patterns = includes if includes else JSONL_GLOB

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type=DATASET_REPO_TYPE,
            local_dir=dest_dir,
            allow_patterns=allow_patterns,
        )
        return True, local_path
    except Exception as e:
        return False, str(e)


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
        "--focus",
        default=None,
        help="Optional directory inside --dir to focus upload and polling",
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
        help="Also download from the HF repo periodically",
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=60.0,
        help="Download sync interval in seconds when --sync is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval_str = str(int(args.check_interval)) if args.check_interval.is_integer() else str(args.check_interval)
    sync_interval_str = str(int(args.sync_interval)) if args.sync_interval.is_integer() else str(args.sync_interval)
    path = Path(args.dir).resolve()

    focus_path = path
    focus_relative = None

    if not path.exists():
        print(f"Error: {args.dir} does not exist.")
        return

    if args.sync_interval <= 0:
        print("Error: --sync-interval must be > 0.")
        return

    try:
        focus_path, focus_relative = _resolve_focus_dir(path, args.focus)
    except ValueError as e:
        print(f"Error: {e}")
        return

    include_patterns = _focus_pattern(focus_relative)

    log("Initial upload...")
    success, out = run_upload(args.dir, args.repo, include_patterns)
    if success:
        log(f"Done. {out}")
    else:
        log(f"Failed: {out}")

    known_files = get_jsonl_file_set(args.dir, focus_path)
    scope = f" in focus directory '{focus_relative}'" if focus_relative and focus_relative != "." else ""
    log(
        f"Tracking {len(known_files)} existing file(s){scope}. Polling every {interval_str}s"
        + (f" Syncing every {sync_interval_str}s." if args.sync else ".")
    )

    sync_interval = args.sync_interval
    last_sync_time = time.time()

    while True:
        time.sleep(args.check_interval)
        current_files = get_jsonl_file_set(args.dir, focus_path)
        new_files = sorted(current_files - known_files)

        now = time.time()
        should_sync = args.sync and (now - last_sync_time) >= sync_interval

        if not new_files and not should_sync:
            continue

        if should_sync:
            log("Syncing from HF...")
            success, out = run_download(args.repo, args.dir, include_patterns)
            if success:
                log(f"Sync done. {out}")
            else:
                log(f"Sync failed: {out}")
            known_files = get_jsonl_file_set(args.dir, focus_path)
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
