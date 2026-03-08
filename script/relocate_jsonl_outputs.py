#!/usr/bin/env python3
"""Move root-level ckpt JSONL outputs into ckpt/{dataset}/{tag}/ folders."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable


def _discover_datasets(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _find_dataset(name: str, datasets: Iterable[str]) -> str | None:
    for dataset in sorted(datasets, key=len, reverse=True):
        token = f"_{dataset}_"
        if token in name:
            return dataset
    return None


def _plan_move(path: Path, datasets: Iterable[str], root: Path) -> Path | None:
    dataset = _find_dataset(path.name, datasets)
    tag_raw: str | None = None
    metric: str | None = None
    seed: str | None = None

    if dataset is not None:
        prefix, rest = path.name.split(f"_{dataset}_", 1)
        match = re.match(r"(?P<metric>(?:maj|pass)\d+)_seed(?P<seed>\d+)\.jsonl$", rest)
        if not match:
            return None
        tag_raw = prefix
        metric = match.group("metric")
        seed = match.group("seed")
    else:
        match = re.match(
            r"^(?P<tag>.+)_(?P<dataset>[A-Za-z0-9]+)_(?P<metric>(?:maj|pass)\d+)_seed(?P<seed>\d+)\.jsonl$",
            path.name,
        )
        if not match:
            return None
        tag_raw = match.group("tag")
        dataset = match.group("dataset")
        metric = match.group("metric")
        seed = match.group("seed")

    tag = tag_raw.replace("_", "-")
    dest_dir = root / dataset / tag
    dest_name = f"{metric}_seed{seed}.jsonl"
    return dest_dir / dest_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relocate root-level ckpt JSONL outputs into ckpt/{dataset}/{tag}/.",
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root ckpt directory (default: ckpt).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves (default: dry-run).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    args = parser.parse_args()

    root = Path(args.ckpt_root)
    if not root.exists():
        raise SystemExit(f"ckpt root not found: {root}")

    datasets = _discover_datasets(root)
    if not datasets:
        print(f"Warning: no dataset folders found under {root}; inferring from filenames.")

    candidates = sorted(root.glob("*.jsonl"))
    if not candidates:
        print(f"No root-level JSONL files found in {root}")
        return

    planned: list[tuple[Path, Path]] = []
    skipped: list[tuple[Path, str]] = []

    for path in candidates:
        dest = _plan_move(path, datasets, root)
        if dest is None:
            skipped.append((path, "unrecognized name format"))
            continue
        if dest.exists() and not args.overwrite:
            skipped.append((path, f"destination exists: {dest}"))
            continue
        planned.append((path, dest))

    for src, dest in planned:
        action = "MOVE" if args.apply else "DRY-RUN"
        print(f"{action}: {src} -> {dest}")
        if args.apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and args.overwrite:
                dest.unlink()
            shutil.move(str(src), str(dest))

    if skipped:
        print("\nSkipped:")
        for path, reason in skipped:
            print(f"  {path}: {reason}")

    print("\nSummary:")
    print(f"  planned: {len(planned)}")
    print(f"  skipped: {len(skipped)}")
    if not args.apply:
        print("  (dry-run) rerun with --apply to move files")


if __name__ == "__main__":
    main()
