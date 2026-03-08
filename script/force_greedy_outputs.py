"""Normalize a greedy JSONL file and clone it across multiple seeds."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

OUTPUT_KEYS = ("response", "completion", "solution", "output", "generated", "answer")
SEED_REGEX = re.compile(r"(seed)(\d+)", re.IGNORECASE)


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


def _parse_csv_args(values: List[str] | None) -> List[str]:
    if not values:
        return []
    items: List[str] = []
    for value in values:
        for row in csv.reader([value], skipinitialspace=True):
            for item in row:
                item = item.strip()
                if item:
                    items.append(item)
    return items


def _resolve_single_input(value: str) -> Path:
    expanded = os.path.expanduser(value)
    if glob.has_magic(expanded):
        matches = sorted(glob.glob(expanded))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {value}")
        if len(matches) > 1:
            raise ValueError(f"Expected a single input file, got {len(matches)}.")
        return Path(matches[0])
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return path


def _extract_output_text(row: Dict[str, Any]) -> str | None:
    for key in OUTPUT_KEYS:
        val = row.get(key)
        if isinstance(val, str):
            return val
    return None


def _extract_solution_text(row: Dict[str, Any]) -> str | None:
    solutions = row.get("solutions")
    if isinstance(solutions, list) and solutions:
        first = solutions[0]
        if isinstance(first, str):
            return first
    return None


def _build_canonical(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    canonical: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(rows, 1):
        pid = _extract_problem_id(row, row_idx)
        entry = canonical.setdefault(
            pid,
            {"output": None, "solution": None, "meta_score": None, "row_score": None},
        )
        if entry["solution"] is None:
            sol = _extract_solution_text(row)
            if sol is not None:
                entry["solution"] = sol
        if entry["output"] is None:
            out = _extract_output_text(row)
            if out is not None:
                entry["output"] = out
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else None
        if entry["meta_score"] is None and meta and "score" in meta:
            entry["meta_score"] = meta["score"]
        if entry["row_score"] is None and "score" in row:
            entry["row_score"] = row["score"]
    return canonical


def _apply_canonical(
    rows: List[Dict[str, Any]],
    canonical: Dict[str, Dict[str, Any]],
    seed_value: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "rows": 0,
        "rows_updated": 0,
        "rows_missing_output": 0,
        "problems_missing_canonical": 0,
    }
    seen_missing = set()
    new_rows: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(rows, 1):
        stats["rows"] += 1
        pid = _extract_problem_id(row, row_idx)
        entry = canonical.get(pid, {})
        canonical_output = entry.get("output") or entry.get("solution")
        canonical_solution = entry.get("solution") or entry.get("output")
        updated = False

        row = dict(row)
        solutions = row.get("solutions")
        if isinstance(solutions, list):
            if canonical_solution is not None:
                row["solutions"] = [canonical_solution for _ in solutions]
                updated = True
            else:
                stats["rows_missing_output"] += 1
                seen_missing.add(pid)
        else:
            if canonical_output is not None:
                touched = False
                for key in OUTPUT_KEYS:
                    if key in row and isinstance(row[key], str):
                        row[key] = canonical_output
                        touched = True
                if touched:
                    updated = True
                else:
                    stats["rows_missing_output"] += 1
                    seen_missing.add(pid)
            else:
                stats["rows_missing_output"] += 1
                seen_missing.add(pid)

        meta = row.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            row["metadata"] = meta
        meta["seed"] = seed_value

        if "score" in meta and entry.get("meta_score") is not None:
            meta["score"] = entry["meta_score"]
            updated = True
        if "score" in row and entry.get("row_score") is not None:
            row["score"] = entry["row_score"]
            updated = True

        if updated:
            stats["rows_updated"] += 1
        new_rows.append(row)

    stats["problems_missing_canonical"] = len(seen_missing)
    return new_rows, stats


def _output_path(input_path: Path, out_dir: Path, seed_value: int) -> Path:
    name = input_path.name
    if SEED_REGEX.search(name):
        name = SEED_REGEX.sub(lambda m: f"{m.group(1)}{seed_value}", name, count=1)
    else:
        name = f"{input_path.stem}_seed{seed_value}{input_path.suffix}"
    return out_dir / name


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a greedy JSONL (per-problem first output) and clone it across seeds."
        )
    )
    parser.add_argument(
        "--inputs",
        required=True,
        action="append",
        help="Single JSONL path/glob (must resolve to exactly one file).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed value (default: 42).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=8,
        help="Number of seeds/files to create (default: 8).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same directory as input).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be written without writing files.",
    )
    args = parser.parse_args()

    input_specs = _parse_csv_args(args.inputs)
    if not input_specs:
        raise RuntimeError("No inputs provided.")
    if len(input_specs) != 1:
        raise ValueError("Provide exactly one input file or glob.")

    input_path = _resolve_single_input(input_specs[0])
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent

    rows = _read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"Input file is empty: {input_path}")
    canonical = _build_canonical(rows)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    print(f"Input: {input_path}")
    print(f"Problems: {len(canonical)}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)")

    for seed_value in seeds:
        new_rows, stats = _apply_canonical(rows, canonical, seed_value)
        output_path = _output_path(input_path, out_dir, seed_value)
        if args.dry_run:
            print(f"{input_path} -> {output_path} (dry-run)")
        else:
            _write_jsonl(output_path, new_rows)
            print(f"{input_path} -> {output_path}")
        print(
            "  rows={rows}, updated={rows_updated}, missing_output_rows={rows_missing_output}, "
            "problems_missing_canonical={problems_missing_canonical}".format(**stats)
        )


if __name__ == "__main__":
    main()
