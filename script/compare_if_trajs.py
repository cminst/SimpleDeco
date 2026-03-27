"""Post-hoc analysis and comparison of IFEval / IFBench trajectory files.

Each input JSONL is produced by ``utils/if_eval.py`` and contains one record
per prompt with per-instruction strict/loose boolean results stored in
``metadata``.  This script aggregates those into the four canonical metrics
and prints a side-by-side comparison table when multiple files are given.

Usage
-----
# Single file — just print metrics
python script/compare_if_trajs.py path/to/run.jsonl

# Multiple files — side-by-side comparison
python script/compare_if_trajs.py run_a.jsonl run_b.jsonl run_c.jsonl

# Glob
python script/compare_if_trajs.py "generation_log/ifeval/*.jsonl"

# Short labels
python script/compare_if_trajs.py run_a.jsonl run_b.jsonl --labels baseline autodeco
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    records = []
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
                records.append(row)
    return records


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_if_metrics(records: List[dict]) -> Dict[str, float]:
    """Return the four canonical IF metrics from a list of saved records.

    Each record must have ``metadata.strict_follow_all``,
    ``metadata.strict_follow_instruction_list``, and their loose equivalents.
    """
    prompt_strict_ok = 0
    prompt_loose_ok = 0
    instr_strict_ok = 0
    instr_strict_total = 0
    instr_loose_ok = 0
    instr_loose_total = 0
    n_prompts = 0

    for rec in records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}

        # strict
        s_all = meta.get("strict_follow_all")
        s_list = meta.get("strict_follow_instruction_list")
        # loose
        l_all = meta.get("loose_follow_all")
        l_list = meta.get("loose_follow_instruction_list")

        if s_all is None or s_list is None or l_all is None or l_list is None:
            raise ValueError(
                "Record is missing per-instruction eval results in metadata. "
                "Make sure the JSONL was produced by utils/if_eval.py (current version)."
            )

        n_prompts += 1
        prompt_strict_ok += int(bool(s_all))
        prompt_loose_ok += int(bool(l_all))
        instr_strict_ok += sum(bool(b) for b in s_list)
        instr_strict_total += len(s_list)
        instr_loose_ok += sum(bool(b) for b in l_list)
        instr_loose_total += len(l_list)

    if n_prompts == 0:
        raise ValueError("No records found in file.")

    ps = prompt_strict_ok / n_prompts
    pl = prompt_loose_ok / n_prompts
    is_ = instr_strict_ok / instr_strict_total if instr_strict_total else 0.0
    il = instr_loose_ok / instr_loose_total if instr_loose_total else 0.0
    avg = (ps + pl + is_ + il) / 4

    return {
        "prompt_strict": ps,
        "prompt_loose": pl,
        "instruction_strict": is_,
        "instruction_loose": il,
        "average": avg,
        "_n_prompts": n_prompts,
        "_n_instructions": instr_strict_total,
    }


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

METRIC_KEYS = ["prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose", "average"]
METRIC_DISPLAY = {
    "prompt_strict": "Prompt Strict",
    "prompt_loose": "Prompt Loose",
    "instruction_strict": "Instr Strict",
    "instruction_loose": "Instr Loose",
    "average": "Average",
}


def _short_label(path: Path) -> str:
    """Derive a compact label from the file path."""
    stem = path.stem
    # strip common suffixes added by if_eval.py
    for suffix in ("-responses", "-eval_strict", "-eval_loose"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    # shorten long stems: keep last two dash-separated tokens if very long
    if len(stem) > 40:
        parts = stem.split("-")
        stem = "-".join(parts[:2]) + "…" + "-".join(parts[-2:]) if len(parts) > 4 else stem[:40] + "…"
    return stem


def _print_single(label: str, metrics: Dict[str, float], path: Path):
    print(f"\nFile : {path}")
    print(f"Label: {label}")
    print(f"  n_prompts     : {int(metrics['_n_prompts'])}")
    print(f"  n_instructions: {int(metrics['_n_instructions'])}")
    print()
    row_label_w = max(len(METRIC_DISPLAY[k]) for k in METRIC_KEYS) + 2
    for k in METRIC_KEYS:
        sep = "─" * 40 if k == "average" else ""
        if sep:
            print(f"  {sep}")
        print(f"  {METRIC_DISPLAY[k]:<{row_label_w}} {metrics[k]:.4f}  ({metrics[k]*100:.2f}%)")


def _print_comparison(
    labels: List[str],
    all_metrics: List[Dict[str, float]],
    paths: List[Path],
    highlight_best: bool = True,
):
    n = len(labels)
    col_w = max(12, max(len(lb) for lb in labels) + 2)
    metric_w = max(len(METRIC_DISPLAY[k]) for k in METRIC_KEYS) + 2

    header = f"{'Metric':<{metric_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for k in METRIC_KEYS:
        values = [m[k] for m in all_metrics]
        best_val = max(values)
        cells = []
        for v in values:
            cell = f"{v:.4f}"
            if highlight_best and abs(v - best_val) < 1e-9 and n > 1:
                cell = f"[{cell}]"
            cells.append(cell)
        if k == "average":
            print("─" * len(header))
        row = f"{METRIC_DISPLAY[k]:<{metric_w}}" + "".join(f"{c:>{col_w}}" for c in cells)
        print(row)
    print(sep)

    print(f"\n{'File':<{metric_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels))
    for i, (lb, p) in enumerate(zip(labels, paths)):
        print(f"  [{i}] {lb}: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc IFEval/IFBench metric report and comparison.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more JSONL files (or glob patterns) saved by utils/if_eval.py.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional short labels for each file (same order as files).",
    )
    parser.add_argument(
        "--no_highlight",
        action="store_true",
        help="Disable [brackets] around best value in comparison table.",
    )
    args = parser.parse_args()

    # Expand globs
    resolved: List[Path] = []
    for pattern in args.files:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            resolved.extend(Path(m) for m in matches)
        else:
            p = Path(pattern)
            if p.exists():
                resolved.append(p)
            else:
                print(f"Warning: no files matched pattern: {pattern}", file=sys.stderr)

    if not resolved:
        parser.error("No input files found.")

    # Deduplicate while preserving order
    seen = set()
    paths: List[Path] = []
    for p in resolved:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(p)

    # Build labels
    if args.labels:
        if len(args.labels) != len(paths):
            parser.error(
                f"--labels has {len(args.labels)} entries but {len(paths)} files were resolved."
            )
        labels = args.labels
    else:
        labels = [_short_label(p) for p in paths]

    # Load + compute
    all_metrics: List[Dict[str, float]] = []
    for p, lb in zip(paths, labels):
        try:
            records = _read_jsonl(p)
        except Exception as exc:
            print(f"Error reading {p}: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            metrics = compute_if_metrics(records)
        except ValueError as exc:
            print(f"Error computing metrics for {p}: {exc}", file=sys.stderr)
            sys.exit(1)
        all_metrics.append(metrics)

    # Report
    if len(paths) == 1:
        _print_single(labels[0], all_metrics[0], paths[0])
    else:
        _print_comparison(labels, all_metrics, paths, highlight_best=not args.no_highlight)

    print()


if __name__ == "__main__":
    main()
