"""Diagnose greedy JSONL files for maj@k/pass@k issues (duplicate samples, IDs, seeds)."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _extract_problem_id(row: Dict[str, Any], fallback_idx: int) -> Tuple[str, str]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("problem_index", "problem_id", "idx", "index"):
        if key in meta:
            return f"{meta[key]}", f"meta:{key}"
        if key in row:
            return f"{row[key]}", f"row:{key}"
    for key in ("problem", "prompt"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return f"{key}:{row[key]}", f"row:{key}"
    return f"row:{fallback_idx}", "fallback:row_index"


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


def _extract_response_text(row: Dict[str, Any]) -> str | None:
    for key in ("response", "completion", "solution", "output", "generated", "answer"):
        if key in row and isinstance(row[key], str):
            return row[key]
    return None


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalized_hash_ignoring_seed(path: Path) -> str:
    h = hashlib.sha256()
    for row in _read_jsonl(path):
        if isinstance(row.get("metadata"), dict):
            meta = dict(row["metadata"])
            for key in list(meta.keys()):
                if "seed" in key.lower():
                    meta.pop(key, None)
            row = dict(row)
            row["metadata"] = meta
        payload = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _summarize_file(path: Path) -> Dict[str, Any]:
    per_problem_samples: Dict[str, int] = {}
    per_problem_rows: Dict[str, int] = {}
    per_problem_sources: Dict[str, str] = {}
    per_problem_unique_outputs: Dict[str, set] = {}
    fallback_ids: List[str] = []
    rows_with_solutions = 0
    solutions_sizes: List[int] = []
    total_rows = 0
    total_samples = 0
    missing_outputs = 0
    seed_keys: Dict[str, set] = {}

    for row_idx, row in enumerate(_read_jsonl(path), 1):
        total_rows += 1
        problem_id, source = _extract_problem_id(row, row_idx)
        per_problem_sources.setdefault(problem_id, source)
        if source.startswith("fallback"):
            fallback_ids.append(problem_id)

        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for key, value in meta.items():
            if "seed" in key.lower():
                seed_keys.setdefault(key, set()).add(value)

        if isinstance(row.get("solutions"), list) and row.get("ground_truth") is not None:
            num_solutions = len(row["solutions"])
            rows_with_solutions += 1
            solutions_sizes.append(num_solutions)
            total_samples += num_solutions
            per_problem_samples[problem_id] = per_problem_samples.get(problem_id, 0) + num_solutions
            per_problem_rows[problem_id] = per_problem_rows.get(problem_id, 0) + 1
            for solution in row["solutions"]:
                if isinstance(solution, str):
                    per_problem_unique_outputs.setdefault(problem_id, set()).add(
                        _hash_text(solution)
                    )
            continue

        total_samples += 1
        per_problem_samples[problem_id] = per_problem_samples.get(problem_id, 0) + 1
        per_problem_rows[problem_id] = per_problem_rows.get(problem_id, 0) + 1
        response = _extract_response_text(row)
        if response is None:
            missing_outputs += 1
        else:
            per_problem_unique_outputs.setdefault(problem_id, set()).add(_hash_text(response))

    multi_sample = {pid: count for pid, count in per_problem_samples.items() if count > 1}
    multi_unique = {
        pid: len(values)
        for pid, values in per_problem_unique_outputs.items()
        if len(values) > 1
    }
    summary = {
        "path": path,
        "rows": total_rows,
        "samples": total_samples,
        "problems": len(per_problem_samples),
        "multi_sample": multi_sample,
        "multi_unique": multi_unique,
        "fallback_ids": fallback_ids,
        "rows_with_solutions": rows_with_solutions,
        "solutions_sizes": solutions_sizes,
        "missing_outputs": missing_outputs,
        "seed_keys": seed_keys,
        "per_problem_samples": per_problem_samples,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose greedy JSONL files for maj@k/pass@k inconsistencies."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        action="append",
        help="Comma-separated list of JSONL paths/globs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ignore-seed",
        action="store_true",
        help="Also compute a normalized hash that ignores metadata keys containing 'seed'.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="How many example problem IDs to show per issue (default: 5).",
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

    print(f"Files: {len(paths)}")

    sizes = {path: path.stat().st_size for path in paths}
    if len(set(sizes.values())) == 1:
        print(f"All file sizes identical: {next(iter(sizes.values()))} bytes")
    else:
        print("File sizes differ:")
        for path, size in sizes.items():
            print(f"  {path}: {size} bytes")

    raw_hashes = {path: _sha256_path(path) for path in paths}
    if len(set(raw_hashes.values())) == 1:
        print("All raw SHA256 hashes identical.")
    else:
        print("Raw SHA256 hashes differ.")

    if args.ignore_seed:
        norm_hashes = {path: _normalized_hash_ignoring_seed(path) for path in paths}
        if len(set(norm_hashes.values())) == 1:
            print("All seed-normalized hashes identical (ignoring metadata *seed* keys).")
        else:
            print("Seed-normalized hashes differ (ignoring metadata *seed* keys).")

    summaries = [_summarize_file(path) for path in paths]

    for summary in summaries:
        path = summary["path"]
        print(f"\n{path}")
        print(
            f"  rows={summary['rows']}, samples={summary['samples']}, "
            f"problems={summary['problems']}"
        )
        multi_sample = summary["multi_sample"]
        if multi_sample:
            print(f"  problems with >1 samples: {len(multi_sample)}")
            for pid in list(multi_sample.keys())[: args.show]:
                print(f"    {pid}: {multi_sample[pid]} samples")
        else:
            print("  problems with >1 samples: 0")

        if summary["rows_with_solutions"]:
            sizes_list = summary["solutions_sizes"]
            if sizes_list:
                min_size = min(sizes_list)
                max_size = max(sizes_list)
                med_size = statistics.median(sizes_list)
                print(
                    f"  rows with solutions list: {summary['rows_with_solutions']} "
                    f"(min/median/max size={min_size}/{med_size}/{max_size})"
                )
            else:
                print(f"  rows with solutions list: {summary['rows_with_solutions']}")
        else:
            print("  rows with solutions list: 0")

        multi_unique = summary["multi_unique"]
        if multi_unique:
            print(f"  problems with >1 unique outputs: {len(multi_unique)}")
            for pid in list(multi_unique.keys())[: args.show]:
                print(f"    {pid}: {multi_unique[pid]} unique outputs")
        else:
            print("  problems with >1 unique outputs: 0")
        if summary["missing_outputs"]:
            print(f"  rows missing output text: {summary['missing_outputs']}")
        else:
            print("  rows missing output text: 0")

        fallback_ids = summary["fallback_ids"]
        if fallback_ids:
            print(f"  fallback problem IDs (row/prompt): {len(set(fallback_ids))}")
            for pid in list(dict.fromkeys(fallback_ids))[: args.show]:
                print(f"    {pid}")
        else:
            print("  fallback problem IDs (row/prompt): 0")

        seed_keys = summary["seed_keys"]
        if seed_keys:
            keys_display = ", ".join(sorted(seed_keys.keys()))
            print(f"  seed metadata keys: {keys_display}")
            for key, values in seed_keys.items():
                values_list = list(values)
                if len(values_list) <= args.show:
                    print(f"    {key}: {values_list}")
                else:
                    print(f"    {key}: {values_list[:args.show]} (+{len(values_list) - args.show} more)")
        else:
            print("  seed metadata keys: none")

    if len(summaries) > 1:
        per_file_counts = [summary["per_problem_samples"] for summary in summaries]
        all_problem_ids = set()
        for counts in per_file_counts:
            all_problem_ids.update(counts.keys())

        differing = []
        not_one = []
        for pid in sorted(all_problem_ids):
            counts = [counts_map.get(pid, 0) for counts_map in per_file_counts]
            if any(c != 1 for c in counts):
                not_one.append((pid, counts))
            if len(set(counts)) > 1:
                differing.append((pid, counts))

        print("\nAcross files")
        print(f"  problems with any count != 1: {len(not_one)} / {len(all_problem_ids)}")
        if not_one:
            for pid, counts in not_one[: args.show]:
                print(f"    {pid}: {counts}")
        print(f"  problems with differing counts across files: {len(differing)}")
        if differing:
            for pid, counts in differing[: args.show]:
                print(f"    {pid}: {counts}")


if __name__ == "__main__":
    main()
