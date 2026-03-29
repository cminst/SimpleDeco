#!/usr/bin/env python3
"""
Convert llm_eval rollout JSONLs into a JSONL dataset with a `messages` column.

The saved rollout JSONLs contain a fully rendered prompt string, which should
not be wrapped again as a chat user message. This converter instead reconstructs
the original user turn from the benchmark source row using
`metadata.dataset` + `metadata.problem_index`, and emits rows like:

  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "chat_template_kwargs": {"reasoning_effort": "medium"},
    ...
  }

The resulting JSONL can be read directly by `script/collect_pertoken_diagnostics.py`
with `--dataset_name <output.jsonl> --assistant_only`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.boxed_extract import normalize_multiple_choice_answer


DEFAULT_FINAL_ANSWER_SUFFIX = "Make sure you output the final answer within \\boxed{}."
MCQ_FINAL_ANSWER_SUFFIX = "Make sure you output the final answer within \\boxed{}."
MCQ_DETAILED_FINAL_ANSWER_SUFFIX = (
    "Put your final letter answer within \\boxed{}, for example \\boxed{A}. "
    "Exactly one answer choice is correct."
)


def model_uses_short_mcq_suffix(model_name_or_path: str) -> bool:
    return "deepseek" in model_name_or_path.lower()


def get_final_answer_suffix(ground_truth: Any, model_name_or_path: str) -> str:
    if normalize_multiple_choice_answer(str(ground_truth)) is not None:
        if model_uses_short_mcq_suffix(model_name_or_path):
            return MCQ_FINAL_ANSWER_SUFFIX
        return MCQ_DETAILED_FINAL_ANSWER_SUFFIX
    return DEFAULT_FINAL_ANSWER_SUFFIX


def build_problem_prompt(problem: str, ground_truth: Any, model_name_or_path: str) -> str:
    return f"{problem.rstrip()}\n{get_final_answer_suffix(ground_truth, model_name_or_path)}"


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
        return [Path(match) for match in matches]
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return [path]


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


def _extract_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _extract_response(row: Dict[str, Any]) -> str | None:
    for key in ("response", "completion", "solution", "output", "generated", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _load_benchmark_rows(dataset_name: str) -> List[Dict[str, Any]]:
    path = _REPO_ROOT / "data" / "TempTest" / f"{dataset_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find benchmark source rows for dataset '{dataset_name}' at {path}."
        )
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


def _build_user_message_content(
    source_row: Dict[str, Any],
    model_name_or_path: str,
) -> str:
    if isinstance(source_row.get("problem"), str):
        if "gt" not in source_row:
            raise KeyError("Benchmark source row has 'problem' but is missing 'gt'.")
        return build_problem_prompt(source_row["problem"], source_row["gt"], model_name_or_path)
    if isinstance(source_row.get("prompt"), str):
        return source_row["prompt"]
    raise KeyError(
        "Benchmark source row does not have a supported prompt field "
        "(expected 'problem' or 'prompt')."
    )


def _infer_tag_name(group_tag: str | None, path: Path) -> str:
    if group_tag:
        return group_tag
    if len(path.parts) >= 2:
        return path.parts[-2]
    return ""


def _infer_dataset_name(group_dataset: str | None, meta: Dict[str, Any], path: Path) -> str:
    if isinstance(meta.get("dataset"), str) and meta["dataset"]:
        return meta["dataset"]
    if group_dataset:
        return group_dataset
    if len(path.parts) >= 3:
        return path.parts[-3]
    return ""


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert llm_eval rollout JSONLs into a local JSONL dataset with a "
            "`messages` column for collect_pertoken_diagnostics.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        action="append",
        help="Comma-separated JSONL paths/globs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name used with --tag/--tags to expand ckpt/{dataset}/{tag}/*.jsonl.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Single tag name to expand under ckpt/{dataset}/{tag}/*.jsonl. Can be repeated.",
    )
    parser.add_argument(
        "--tags",
        action="append",
        dest="tags",
        help="Comma-separated tag names to expand under ckpt/{dataset}/{tag}/*.jsonl.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        help="Optional comma-separated labels for the input groups (same order as inputs/tags).",
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root directory for --dataset/--tag(s) expansion.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSONL path for the converted messages dataset.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of converted rollout rows.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress to stderr.",
    )
    args = parser.parse_args()

    input_specs = _parse_csv_args(args.inputs)
    input_groups: List[Dict[str, str]] = []
    for spec in input_specs:
        input_groups.append({"spec": spec, "label": spec, "dataset": args.dataset or "", "tag": ""})

    tag_specs = _parse_csv_args(args.tags)
    if tag_specs:
        if not args.dataset:
            raise ValueError("--dataset is required when using --tag/--tags.")
        for tag in tag_specs:
            spec = os.path.join(args.ckpt_root, args.dataset, tag, "*.jsonl")
            input_groups.append({"spec": spec, "label": tag, "dataset": args.dataset, "tag": tag})

    if not input_groups:
        raise RuntimeError("No inputs provided. Use --inputs or --dataset with --tag/--tags.")

    label_specs = _parse_csv_args(args.labels)
    if label_specs and len(label_specs) != len(input_groups):
        raise ValueError("Number of labels must match number of input groups.")
    if label_specs:
        for group, label in zip(input_groups, label_specs):
            group["label"] = label

    resolved_groups: List[Dict[str, Any]] = []
    for group in input_groups:
        resolved_groups.append({**group, "paths": _resolve_inputs(group["spec"])})

    benchmark_cache: Dict[str, List[Dict[str, Any]]] = {}
    converted = 0
    skipped_missing_response = 0
    skipped_missing_dataset = 0
    skipped_bad_index = 0

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for group in resolved_groups:
            _log(args.progress, f"[group] {group['label']} -> {len(group['paths'])} file(s)")
            for path in group["paths"]:
                _log(args.progress, f"[file] {path}")
                for row in _read_jsonl(path):
                    if args.max_examples is not None and converted >= args.max_examples:
                        break

                    response = _extract_response(row)
                    if response is None:
                        skipped_missing_response += 1
                        continue

                    meta = _extract_metadata(row)
                    dataset_name = _infer_dataset_name(group["dataset"], meta, path)
                    if not dataset_name:
                        skipped_missing_dataset += 1
                        continue

                    if dataset_name not in benchmark_cache:
                        benchmark_cache[dataset_name] = _load_benchmark_rows(dataset_name)
                    benchmark_rows = benchmark_cache[dataset_name]

                    problem_index = _as_int(meta.get("problem_index"))
                    if problem_index < 0 or problem_index >= len(benchmark_rows):
                        skipped_bad_index += 1
                        continue

                    source_row = benchmark_rows[problem_index]
                    model_name_or_path = str(meta.get("model_name_or_path") or "")
                    user_content = _build_user_message_content(source_row, model_name_or_path)

                    converted_row: Dict[str, Any] = {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": response},
                        ],
                        "rollout_label": group["label"],
                        "rollout_dataset": dataset_name,
                        "rollout_tag": _infer_tag_name(group["tag"], path),
                        "source_file": path.name,
                        "source_path": str(path),
                        "problem_index": problem_index,
                        "sample_index": _as_int(meta.get("sample_index")),
                        "seed": _as_int(meta.get("seed")),
                        "score": meta.get("score"),
                        "ground_truth": meta.get("ground_truth", source_row.get("gt")),
                        "rendered_prompt": row.get("prompt"),
                    }

                    reasoning_effort = meta.get("reasoning_effort")
                    if isinstance(reasoning_effort, str) and reasoning_effort:
                        converted_row["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}

                    out_f.write(json.dumps(converted_row, ensure_ascii=False) + "\n")
                    converted += 1

                if args.max_examples is not None and converted >= args.max_examples:
                    break
            if args.max_examples is not None and converted >= args.max_examples:
                break

    metadata = {
        "output_path": str(output_path),
        "converted_rows": int(converted),
        "skipped_missing_response": int(skipped_missing_response),
        "skipped_missing_dataset": int(skipped_missing_dataset),
        "skipped_bad_index": int(skipped_bad_index),
        "input_groups": [
            {"label": group["label"], "spec": group["spec"], "dataset": group["dataset"], "tag": group["tag"]}
            for group in input_groups
        ],
        "recommended_collect_command": (
            "python3 script/collect_pertoken_diagnostics.py "
            f"--dataset_name {str(output_path)!r} "
            "--dataset_text_field messages "
            "--assistant_only "
            "--enable_thinking "
            "--output-dir <OUT_DIR> "
            "--model_name_or_path <MODEL>"
        ),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"[convert_rollouts_to_messages_dataset] Wrote {converted:,} row(s) to {output_path} "
        f"(skipped missing_response={skipped_missing_response}, missing_dataset={skipped_missing_dataset}, "
        f"bad_index={skipped_bad_index})"
    )
    print(f"[convert_rollouts_to_messages_dataset] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
