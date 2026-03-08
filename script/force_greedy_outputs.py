"""Force each problem's outputs in JSONL files to match the first sample."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

OUTPUT_KEYS = ("response", "completion", "solution", "output", "generated", "answer")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                yield row


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


def _resolve_inputs(value: str) -> List[Path]:
    expanded = os.path.expanduser(value)
    if glob.has_magic(expanded):
        matches = sorted(glob.glob(expanded))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {value}")
        return [Path(m) for m in matches]
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return [path]


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


def _canonical_for_files(path: Path) -> Dict[str, Dict[str, Any]]:
    canonical: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(_read_jsonl(path), 1):
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


def _write_forced(
    path: Path,
    output_path: Path,
    canonical: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    stats = {
        "rows": 0,
        "rows_updated": 0,
        "rows_missing_output": 0,
        "problems_missing_canonical": 0,
    }
    seen_missing = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(_read_jsonl(path), 1):
            stats["rows"] += 1
            pid = _extract_problem_id(row, row_idx)
            entry = canonical.get(pid, {})
            canonical_output = entry.get("output") or entry.get("solution")
            canonical_solution = entry.get("solution") or entry.get("output")
            updated = False

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
                    if not touched:
                        stats["rows_missing_output"] += 1
                        seen_missing.add(pid)
                    else:
                        updated = True
                else:
                    stats["rows_missing_output"] += 1
                    seen_missing.add(pid)

            meta = row.get("metadata")
            if isinstance(meta, dict) and "score" in meta and entry.get("meta_score") is not None:
                meta["score"] = entry["meta_score"]
                updated = True
            if "score" in row and entry.get("row_score") is not None:
                row["score"] = entry["row_score"]
                updated = True

            if updated:
                stats["rows_updated"] += 1
            out.write(json.dumps(row, ensure_ascii=True))
            out.write("\n")

    stats["problems_missing_canonical"] = len(seen_missing)
    return stats


def _build_output_path(path: Path, out_dir: Path | None, suffix: str) -> Path:
    if out_dir is None:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return out_dir / f"{path.stem}{suffix}{path.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Force per-problem outputs to match the first sample, making files greedy-like."
        )
    )
    parser.add_argument(
        "--inputs",
        required=True,
        action="append",
        help="Comma-separated list of JSONL paths/globs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write modified files (default: alongside inputs).",
    )
    parser.add_argument(
        "--suffix",
        default="_greedy",
        help="Suffix to append before the file extension (default: _greedy).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite inputs in-place (use with care).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change without writing files.",
    )
    args = parser.parse_args()

    input_specs = _parse_csv_args(args.inputs)
    if not input_specs:
        raise RuntimeError("No inputs provided.")

    paths: List[Path] = []
    for spec in input_specs:
        paths.extend(_resolve_inputs(spec))

    if not paths:
        raise RuntimeError("No JSONL files matched the inputs.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.inplace and out_dir is not None:
        raise ValueError("--inplace cannot be combined with --out-dir.")

    for path in paths:
        canonical = _canonical_for_files(path)
        output_path = path if args.inplace else _build_output_path(path, out_dir, args.suffix)
        if args.dry_run:
            print(f"{path} -> {output_path} (dry-run)")
            print(f"  problems: {len(canonical)}")
            missing = [pid for pid, entry in canonical.items() if not (entry["output"] or entry["solution"])]
            print(f"  problems missing canonical output: {len(missing)}")
            continue
        stats = _write_forced(path, output_path, canonical)
        print(f"{path} -> {output_path}")
        print(
            "  rows={rows}, updated={rows_updated}, missing_output_rows={rows_missing_output}, "
            "problems_missing_canonical={problems_missing_canonical}".format(**stats)
        )


if __name__ == "__main__":
    main()
