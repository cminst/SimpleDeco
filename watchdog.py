import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from huggingface_hub import HfApi, snapshot_download


DATASET_REPO_TYPE = "dataset"
JSONL_GLOB = "**/*.jsonl"


@dataclass(frozen=True)
class FocusFilters:
    focus_dirs: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def has_filters(self) -> bool:
        return bool(self.focus_dirs or self.datasets or self.tags)

    def matches_relative_path(self, relative_path: str) -> bool:
        path = PurePosixPath(relative_path)
        if path.suffix != ".jsonl":
            return False

        parts = path.parts
        dataset = parts[0] if len(parts) >= 2 else None
        tag = parts[1] if len(parts) >= 3 else None

        if self.focus_dirs and not any(_is_under_focus(relative_path, focus_dir) for focus_dir in self.focus_dirs):
            return False
        if self.datasets and dataset not in self.datasets:
            return False
        if self.tags and tag not in self.tags:
            return False
        return True

    def describe(self) -> str:
        pieces = []
        if self.focus_dirs:
            pieces.append(f"focus={', '.join(self.focus_dirs)}")
        if self.datasets:
            pieces.append(f"datasets={', '.join(self.datasets)}")
        if self.tags:
            pieces.append(f"tags={', '.join(self.tags)}")
        return "; ".join(pieces) if pieces else "all files"


def _is_under_focus(relative_path: str, focus_dir: str) -> bool:
    if focus_dir in ("", "."):
        return True
    return relative_path == focus_dir or relative_path.startswith(f"{focus_dir}/")


def _parse_multi_values(values: list[str] | None) -> tuple[str, ...]:
    parsed: list[str] = []
    seen: set[str] = set()

    for value in values or []:
        for item in value.split(","):
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            parsed.append(cleaned)
            seen.add(cleaned)

    return tuple(parsed)


def _normalize_focus_dir(directory_to_watch: Path, focus: str) -> str:
    focused = Path(focus)
    if not focused.is_absolute():
        focused = directory_to_watch / focused

    focused = focused.resolve(strict=False)

    try:
        focus_relative = focused.relative_to(directory_to_watch).as_posix()
    except ValueError as exc:
        raise ValueError(f"Focus directory '{focus}' must be inside --dir '{directory_to_watch}'.") from exc

    return focus_relative


def _build_filters(directory_to_watch: Path, focus_args: list[str] | None, dataset_args: list[str] | None,
                   tag_args: list[str] | None) -> FocusFilters:
    focus_dirs = tuple(
        focus_relative
        for focus in _parse_multi_values(focus_args)
        if (focus_relative := _normalize_focus_dir(directory_to_watch, focus)) not in ("", ".")
    )
    datasets = _parse_multi_values(dataset_args)
    tags = _parse_multi_values(tag_args)
    return FocusFilters(focus_dirs=focus_dirs, datasets=datasets, tags=tags)


def get_jsonl_file_set(path: str, filters: FocusFilters | None = None) -> set[str]:
    root = Path(path).resolve()
    if not root.exists():
        return set()

    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.jsonl")
        if p.is_file() and (filters is None or filters.matches_relative_path(p.relative_to(root).as_posix()))
    }


def list_remote_jsonl_files(api: HfApi, repo_id: str, filters: FocusFilters | None = None) -> list[str]:
    return sorted(
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type=DATASET_REPO_TYPE)
        if path.endswith(".jsonl") and (filters is None or filters.matches_relative_path(path))
    )


def run_upload(api: HfApi, directory_to_watch: str, repo_id: str, includes: list[str] | None = None):
    if includes == []:
        return True, "No matching local files to upload."

    allow_patterns = includes if includes is not None else JSONL_GLOB

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
    if includes == []:
        return True, "No matching remote files to download."

    allow_patterns = includes if includes is not None else JSONL_GLOB

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
        action="append",
        default=None,
        help="Optional subdirectory inside --dir to focus on. Repeat or use commas for multiple values.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        action="append",
        default=None,
        help="Dataset name(s) to match under --dir. Repeat or use commas for multiple values.",
    )
    parser.add_argument(
        "--tag",
        "--tags",
        dest="tags",
        action="append",
        default=None,
        help="Tag name(s) to match under each dataset directory. Repeat or use commas for multiple values.",
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

    if not path.exists():
        print(f"Error: {args.dir} does not exist.")
        return

    if args.sync_interval <= 0:
        print("Error: --sync-interval must be > 0.")
        return

    api = HfApi()

    try:
        filters = _build_filters(path, args.focus, args.datasets, args.tags)
    except ValueError as e:
        print(f"Error: {e}")
        return

    initial_upload_files = sorted(get_jsonl_file_set(args.dir, filters)) if filters.has_filters() else None

    log("Initial upload...")
    success, out = run_upload(api, args.dir, args.repo, initial_upload_files)
    if success:
        log(f"Done. {out}")
    else:
        log(f"Failed: {out}")

    known_files = get_jsonl_file_set(args.dir, filters)
    scope = f" with filters [{filters.describe()}]" if filters.has_filters() else ""
    log(
        f"Tracking {len(known_files)} existing file(s){scope}. Polling every {interval_str}s"
        + (f" Syncing every {sync_interval_str}s." if args.sync else ".")
    )

    sync_interval = args.sync_interval
    last_sync_time = time.time()

    while True:
        time.sleep(args.check_interval)
        current_files = get_jsonl_file_set(args.dir, filters)
        new_files = sorted(current_files - known_files)

        now = time.time()
        should_sync = args.sync and (now - last_sync_time) >= sync_interval

        if not new_files and not should_sync:
            continue

        if should_sync:
            log("Syncing from HF...")
            remote_files = list_remote_jsonl_files(api, args.repo, filters) if filters.has_filters() else None
            success, out = run_download(args.repo, args.dir, remote_files)
            if success:
                log(f"Sync done. {out}")
            else:
                log(f"Sync failed: {out}")
            known_files = get_jsonl_file_set(args.dir, filters)
            last_sync_time = time.time()

        if not new_files:
            continue

        log(
            f"{len(new_files)} new file(s): {', '.join(new_files[:5])}"
            + (f" ... and {len(new_files)-5} more" if len(new_files) > 5 else "")
        )

        success, out = run_upload(api, args.dir, args.repo, new_files)
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
