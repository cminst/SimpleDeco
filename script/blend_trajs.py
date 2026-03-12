#!/usr/bin/env python3
"""Blend two sets of trajectory JSONLs by mixing rows at the sample level.

The output keeps the base seed set, but for each sample row we optionally
swap in the corresponding row from the alt tag (matched by problem + sample).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} in {path}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_problem_id(row: Dict[str, Any], fallback_idx: int) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("problem_index", "problem_id", "idx", "index"):
        if key in meta:
            return f"{meta[key]}"
        if key in row:
            return f"{row[key]}"
    for key in ("problem", "prompt"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return f"{key}:{row[key]}"
    return f"row:{fallback_idx}"


def _extract_sample_index(row: Dict[str, Any], fallback_idx: int) -> int:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("sample_index", "sample_id", "sample_idx"):
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                break
        if key in row:
            try:
                return int(row[key])
            except Exception:
                break
    return fallback_idx


def _index_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    per_problem_counter: Dict[str, int] = defaultdict(int)
    indexed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row_idx, row in enumerate(rows):
        pid = _extract_problem_id(row, row_idx)
        sample_idx_default = per_problem_counter[pid]
        sample_idx = _extract_sample_index(row, sample_idx_default)
        per_problem_counter[pid] += 1
        indexed[(pid, sample_idx)] = row
        indexed.setdefault((pid, sample_idx_default), row)
    return indexed


def _blend_rows(
    base_rows: List[Dict[str, Any]],
    alt_index: Dict[Tuple[str, int], Dict[str, Any]],
    mix_prob: float,
    rng_seed: int,
) -> List[Dict[str, Any]]:
    import random

    rng = random.Random(rng_seed)
    out: List[Dict[str, Any]] = []
    per_problem_counter: Dict[str, int] = defaultdict(int)
    swapped = 0
    for row_idx, row in enumerate(base_rows):
        pid = _extract_problem_id(row, row_idx)
        sample_idx_default = per_problem_counter[pid]
        sample_idx = _extract_sample_index(row, sample_idx_default)
        per_problem_counter[pid] += 1
        alt_row = alt_index.get((pid, sample_idx))
        if alt_row is None:
            alt_row = alt_index.get((pid, sample_idx_default))
        if alt_row is not None and rng.random() < mix_prob:
            out.append(alt_row)
            swapped += 1
        else:
            out.append(row)
    return out


def _discover_seed_files(dataset: str, tag: str, num_samples: int | None) -> Dict[int, Path]:
    if num_samples is None:
        pattern = f"ckpt/{dataset}/{tag}/maj*_seed*.jsonl"
    else:
        pattern = f"ckpt/{dataset}/{tag}/maj{num_samples}_seed*.jsonl"
    files = sorted(Path().glob(pattern))
    seed_files: Dict[int, Path] = {}
    for path in files:
        match = re.search(r"maj(\d+)_seed(\d+)\.jsonl$", path.name)
        if not match:
            continue
        if num_samples is not None and int(match.group(1)) != num_samples:
            continue
        seed_files[int(match.group(2))] = path
    return seed_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend two trajectory tags into a new tag.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--alt-tag", required=True)
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    if not (0.0 <= args.mix_prob <= 1.0):
        raise SystemExit("--mix-prob must be between 0 and 1.")

    base_files = _discover_seed_files(args.dataset, args.base_tag, args.num_samples)
    alt_files = _discover_seed_files(args.dataset, args.alt_tag, args.num_samples)
    if not base_files:
        raise SystemExit(f"No base files found for {args.base_tag}.")
    if not alt_files:
        raise SystemExit(f"No alt files found for {args.alt_tag}.")

    base_seeds = sorted(base_files.keys())
    alt_seeds = sorted(alt_files.keys())

    alt_cache: Dict[int, List[Dict[str, Any]]] = {}
    for idx, seed in enumerate(base_seeds):
        alt_seed = alt_seeds[idx % len(alt_seeds)]
        base_path = base_files[seed]
        alt_path = alt_files[alt_seed]

        base_rows = _read_jsonl(base_path)
        if alt_seed not in alt_cache:
            alt_cache[alt_seed] = _read_jsonl(alt_path)
        alt_rows = alt_cache[alt_seed]

        alt_index = _index_rows(alt_rows)
        blended = _blend_rows(
            base_rows,
            alt_index,
            args.mix_prob,
            rng_seed=seed + args.seed_offset,
        )

        out_path = Path(
            f"ckpt/{args.dataset}/{args.out_tag}/maj{args.num_samples}_seed{seed}.jsonl"
        )
        _write_jsonl(out_path, blended)
        print(
            f"Blended seed {seed} with alt seed {alt_seed} -> {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
