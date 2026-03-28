"""Post-hoc analysis and comparison of IFEval / IFBench trajectory files.

Each input JSONL is produced by ``utils/if_eval.py`` and contains one record
per prompt with per-instruction strict/loose boolean results stored in
``metadata``. This script aggregates those into the four canonical metrics and
can compare either individual JSONLs or groups of JSONLs (for example, all
seed files under one tag).

Usage
-----
# Single file — just print metrics
python script/compare_if_trajs.py path/to/run.jsonl

# Multiple files — side-by-side comparison
python script/compare_if_trajs.py run_a.jsonl run_b.jsonl run_c.jsonl

# Group inputs with compare_trajs.py-style flags
python script/compare_if_trajs.py --dataset ifeval --tags base-r1-distill-qwen7b,autodeco-r1-distill-qwen7b

# Optional labels
python script/compare_if_trajs.py --dataset ifeval --tags base-r1-distill-qwen7b,autodeco-r1-distill-qwen7b \
  --labels baseline,autodeco
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _t_critical_975(df: int) -> float:
    if df <= 0:
        return 0.0
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


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


def _resolve_inputs(value: str) -> List[Path]:
    expanded = os.path.expanduser(value)
    if glob.has_magic(expanded):
        matches = sorted(glob.glob(expanded))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {value}")
        return [Path(match) for match in matches]
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return [path]


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _default_group_label(spec: str, paths: List[Path]) -> str:
    if len(paths) == 1:
        return _short_label(paths[0])
    parent_names = {path.parent.name for path in paths}
    if len(parent_names) == 1:
        return next(iter(parent_names))
    return spec


def _infer_dataset_name(spec: str, paths: List[Path]) -> str | None:
    parts = Path(spec).parts
    if "ckpt" in parts:
        ckpt_idx = parts.index("ckpt")
        if ckpt_idx + 1 < len(parts):
            return parts[ckpt_idx + 1]
    for path in paths:
        if "ckpt" in path.parts:
            parts = path.parts
            ckpt_idx = parts.index("ckpt")
            if ckpt_idx + 1 < len(parts):
                return parts[ckpt_idx + 1]
    return None


def _truncate_paths(paths: List[Path], max_items: int = 5) -> List[Path | str]:
    if len(paths) <= max_items:
        return paths
    if max_items < 5:
        return paths[:max_items]
    head_count = 2
    tail_count = max_items - head_count - 1
    return paths[:head_count] + ["..."] + paths[-tail_count:]


def _summarize_group(paths: List[Path]) -> Dict[str, object]:
    file_metrics: List[Dict[str, float]] = []
    for path in paths:
        records = _read_jsonl(path)
        file_metrics.append(compute_if_metrics(records))

    # Mean of per-seed means; 95% CI via t-distribution when >1 seed.
    mean_metrics: Dict[str, float] = {}
    ci_metrics: Dict[str, Optional[float]] = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in file_metrics]
        mean_metrics[key] = sum(vals) / len(vals)
        if len(vals) > 1:
            stdev = statistics.stdev(vals)
            t_crit = _t_critical_975(len(vals) - 1)
            ci_metrics[key] = t_crit * stdev / math.sqrt(len(vals))
        else:
            ci_metrics[key] = None

    return {
        "metrics": mean_metrics,
        "metric_ci": ci_metrics,
        "n_files": len(paths),
        "prompt_counts": [int(m["_n_prompts"]) for m in file_metrics],
        "instruction_counts": [int(m["_n_instructions"]) for m in file_metrics],
    }


def _format_count_summary(values: List[int]) -> str:
    unique = sorted(set(values))
    if not unique:
        return "0"
    if len(unique) == 1:
        return str(unique[0])
    return f"{unique[0]}-{unique[-1]}"


def _format_metric(value: float, ci: Optional[float]) -> str:
    if ci is None:
        return f"{value * 100:.2f}%"
    return f"{value * 100:.2f}±{ci * 100:.2f}%"


def _bold_text(value: str) -> str:
    return f"\033[1m{value}\033[0m"


def _print_single(
    label: str,
    summary: Dict[str, object],
    paths: List[Path],
    spec: str,
    dataset: str | None,
):
    kind = "File" if len(paths) == 1 else "Group"
    print(f"\n{kind} : {paths[0] if len(paths) == 1 else spec}")
    print(f"Label: {label}")
    if dataset:
        print(f"Dataset: {dataset}")
    if len(paths) > 1:
        print(f"Files: {len(paths)}")
    print(f"  n_prompts     : {_format_count_summary(summary['prompt_counts'])}")
    print(f"  n_instructions: {_format_count_summary(summary['instruction_counts'])}")
    print()
    row_label_w = max(len(METRIC_DISPLAY[k]) for k in METRIC_KEYS) + 2
    metrics = summary["metrics"]
    metric_ci = summary["metric_ci"]
    for k in METRIC_KEYS:
        if k == "average":
            print(f"  {'─' * 40}")
        value = metrics[k]
        ci = metric_ci[k]
        print(f"  {METRIC_DISPLAY[k]:<{row_label_w}} {_format_metric(value, ci)}")

    if len(paths) > 1:
        print("\nMatched files:")
        shown = _truncate_paths(paths)
        for item in shown:
            if item == "...":
                print("       ...")
                continue
            idx = paths.index(item)
            print(f"  [{idx}] {item}")


def _print_comparison(
    groups: List[Dict[str, object]],
    highlight_best: bool = True,
):
    labels = [str(group["label"]) for group in groups]
    n = len(labels)
    col_w = max(
        12,
        max(len(label) for label in labels) + 2,
        max(
            len(_format_metric(group["metrics"][key], group["metric_ci"][key]))
            for group in groups
            for key in METRIC_KEYS
        ) + 2,
    )
    metric_w = max(len(METRIC_DISPLAY[k]) for k in METRIC_KEYS) + 2
    dataset_values = [group.get("dataset") for group in groups]
    common_dataset = None
    if all(dataset_values) and len(set(dataset_values)) == 1:
        common_dataset = str(dataset_values[0])
    metric_label = "Metric" if common_dataset is None else f"Metric ({common_dataset})"
    metric_label = (
        "Metric" if common_dataset is None else f"Metric ({_bold_text(common_dataset)})"
    )
    metric_w = max(metric_w, len(f"Metric ({common_dataset})" if common_dataset else metric_label) + 2)

    header = f"{metric_label:<{metric_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for k in METRIC_KEYS:
        values = [group["metrics"][k] for group in groups]
        best_val = max(values)
        cells = []
        for group in groups:
            value = group["metrics"][k]
            ci = group["metric_ci"][k]
            cell = _format_metric(value, ci)
            if highlight_best and abs(value - best_val) < 1e-9 and n > 1:
                cell = f"[{cell}]"
            cells.append(cell)
        if k == "average":
            print("─" * len(header))
        row = f"{METRIC_DISPLAY[k]:<{metric_w}}" + "".join(f"{c:>{col_w}}" for c in cells)
        print(row)
    print(sep)

    metadata_rows = [
        ("Files", [str(group["summary"]["n_files"]) for group in groups]),
        ("Prompts", [_format_count_summary(group["summary"]["prompt_counts"]) for group in groups]),
        (
            "Instructions",
            [_format_count_summary(group["summary"]["instruction_counts"]) for group in groups],
        ),
    ]
    for title, cells in metadata_rows:
        print(f"{title:<{metric_w}}" + "".join(f"{cell:>{col_w}}" for cell in cells))

    print(f"\n{'Source':<{metric_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels))
    for i, group in enumerate(groups):
        paths = group["paths"]
        if len(paths) == 1:
            print(f"  [{i}] {group['label']}: {paths[0]}")
            continue
        print(f"  [{i}] {group['label']}: {len(paths)} files from {group['spec']}")
        shown = _truncate_paths(paths)
        for item in shown:
            if item == "...":
                print("       ...")
                continue
            idx = paths.index(item)
            print(f"       [{idx}] {item}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc IFEval/IFBench metric report and comparison. Accepts either "
            "explicit JSONL paths/globs or compare_trajs.py-style --dataset/--tags inputs."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional JSONL files or glob patterns saved by utils/if_eval.py.",
    )
    parser.add_argument(
        "--inputs",
        action="extend",
        nargs="+",
        help="JSONL paths/globs. Accepts comma-separated values and can be passed multiple times.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        action="extend",
        nargs="+",
        help="Labels for input groups. Accepts comma-separated values and can be passed multiple times.",
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root directory for --dataset/--tags expansion (default: ckpt).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name to expand --tags into ckpt/{dataset}/{tag}/*.jsonl.",
    )
    parser.add_argument(
        "--tags",
        action="extend",
        nargs="+",
        help="Tag names to compare. Accepts comma-separated values and can be passed multiple times.",
    )
    parser.add_argument(
        "--no_highlight",
        action="store_true",
        help="Disable [brackets] around best value in comparison table.",
    )
    args = parser.parse_args()

    input_groups: List[Dict[str, str | None]] = [{"spec": spec, "label": None} for spec in args.files]
    input_specs = _parse_csv_args(args.inputs)
    input_groups.extend({"spec": spec, "label": None} for spec in input_specs)

    tag_specs = _parse_csv_args(args.tags)
    if tag_specs:
        if not args.dataset:
            parser.error("--dataset is required when using --tags.")
        for tag in tag_specs:
            input_groups.append(
                {
                    "spec": os.path.join(args.ckpt_root, args.dataset, tag, "*.jsonl"),
                    "label": tag,
                }
            )

    if not input_groups:
        parser.error("No inputs provided. Use positional files, --inputs, or --dataset/--tags.")

    label_specs = _parse_csv_args(args.labels)
    if label_specs and len(label_specs) != len(input_groups):
        parser.error(
            f"--labels has {len(label_specs)} entries but there are {len(input_groups)} input groups."
        )
    if label_specs:
        for group, label in zip(input_groups, label_specs):
            group["label"] = label

    groups: List[Dict[str, object]] = []
    for group in input_groups:
        spec = str(group["spec"])
        try:
            paths = _dedupe_paths(_resolve_inputs(spec))
        except Exception as exc:
            print(f"Error resolving {spec}: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            summary = _summarize_group(paths)
        except Exception as exc:
            print(f"Error reading {spec}: {exc}", file=sys.stderr)
            sys.exit(1)
        label = str(group["label"] or _default_group_label(spec, paths))
        dataset = _infer_dataset_name(spec, paths)
        groups.append(
            {
                "spec": spec,
                "label": label,
                "paths": paths,
                "dataset": dataset,
                "summary": summary,
                "metrics": summary["metrics"],
                "metric_ci": summary["metric_ci"],
            }
        )

    # Report
    if len(groups) == 1:
        group = groups[0]
        _print_single(
            str(group["label"]),
            group["summary"],
            group["paths"],
            str(group["spec"]),
            group.get("dataset"),
        )
    else:
        _print_comparison(groups, highlight_best=not args.no_highlight)

    print()


if __name__ == "__main__":
    main()
