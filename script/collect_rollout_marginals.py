#!/usr/bin/env python3
"""
Collect teacher-forced logit-shape marginals on llm_eval rollout JSONLs.

This script reads free-generation outputs saved by utils/llm_eval.py,
reconstructs prompt + response sequences, and runs teacher forcing on the
response tokens only. It stores a HuggingFace DatasetDict with:

  - tokens: per-token entropy, normalized entropy, top-1 confidence, and
    q-weighted logit variance
  - sequences: per-rollout metadata (dataset/tag/file/problem/seed/score/etc.)

It also writes compact summary files with overall and per-tag marginal stats.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _extract_scalar_text(row: Dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _infer_dataset_name(group_dataset: str | None, meta: Dict[str, Any], path: Path) -> str:
    if isinstance(meta.get("dataset"), str) and meta["dataset"]:
        return meta["dataset"]
    if group_dataset:
        return group_dataset
    if len(path.parts) >= 3:
        return path.parts[-3]
    return ""


def _infer_tag_name(group_tag: str | None, path: Path) -> str:
    if group_tag:
        return group_tag
    if len(path.parts) >= 2:
        return path.parts[-2]
    return ""


def _tokenize_prompt_response(
    prompt_text: str,
    response_text: str,
    tokenizer: Any,
    max_seq_len: int | None,
) -> tuple[List[int], List[int], int, int, int, bool]:
    prompt_ids = tokenizer(text=prompt_text)["input_ids"]
    input_ids = tokenizer(text=prompt_text + response_text)["input_ids"]

    if tokenizer.bos_token_id is not None:
        if prompt_ids and prompt_ids[0] == tokenizer.bos_token_id:
            prompt_ids = prompt_ids[1:]
        if input_ids and input_ids[0] == tokenizer.bos_token_id:
            input_ids = input_ids[1:]

    prompt_len = len(prompt_ids)
    response_len = max(len(input_ids) - prompt_len, 0)
    label_mask = [0] * prompt_len + [1] * response_len
    response_offset = 0
    was_truncated = False

    if max_seq_len is not None and max_seq_len > 0 and len(input_ids) > max_seq_len:
        orig_response_len = response_len
        input_ids = input_ids[-max_seq_len:]
        label_mask = label_mask[-max_seq_len:]
        response_len = int(sum(label_mask))
        prompt_len = len(label_mask) - response_len
        response_offset = max(orig_response_len - response_len, 0)
        was_truncated = True

    return input_ids, label_mask, prompt_len, response_len, response_offset, was_truncated


def _pad_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(example["input_ids"]) for example in batch)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    label_mask: List[List[int]] = []
    for example in batch:
        ids = example["input_ids"]
        mask = example["label_mask"]
        padding = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        label_mask.append(mask + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label_mask": torch.tensor(label_mask, dtype=torch.long),
    }


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def _load_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str | None,
    trust_remote_code: bool,
) -> Any:
    errors: List[str] = []
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(f"AutoModelForCausalLM: {exc}")

    try:
        from model.templlm_auto import AutoDecoModelForCausalLM  # type: ignore
    except Exception:
        AutoDecoModelForCausalLM = None

    if AutoDecoModelForCausalLM is not None:
        try:
            return AutoDecoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(f"AutoDecoModelForCausalLM: {exc}")

    raise RuntimeError(
        "Failed to load model for teacher-forced rollout stats.\n"
        + "\n".join(errors)
    )


def _metric_summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _summarize_token_dataset(tokens_dataset: Dataset) -> Dict[str, Any]:
    cols = tokens_dataset.select_columns(["label", "H_norm", "logit_variance", "p_max"]).with_format("numpy")[:]
    labels = np.asarray(cols["label"])
    h_norm = np.asarray(cols["H_norm"], dtype=np.float64)
    logit_variance = np.asarray(cols["logit_variance"], dtype=np.float64)
    p_max = np.asarray(cols["p_max"], dtype=np.float64)

    def make_payload(mask: np.ndarray) -> Dict[str, Any]:
        return {
            "token_count": int(np.sum(mask)),
            "H_norm": _metric_summary(h_norm[mask]),
            "logit_variance": _metric_summary(logit_variance[mask]),
            "p_max": _metric_summary(p_max[mask]),
        }

    overall_mask = np.ones(labels.shape[0], dtype=bool)
    summary = {
        "overall": make_payload(overall_mask),
        "by_label": {},
    }
    for label in sorted({str(x) for x in labels.tolist()}):
        mask = labels == label
        summary["by_label"][label] = make_payload(mask)
    return summary


def _format_summary_text(summary: Dict[str, Any]) -> str:
    lines = ["Teacher-forced rollout marginals summary", ""]
    overall = summary["overall"]
    lines.append(f"Overall tokens: {overall['token_count']:,}")
    for metric_name in ("H_norm", "logit_variance", "p_max"):
        metric = overall[metric_name]
        lines.append(
            f"  {metric_name}: mean={metric['mean']:.6f} std={metric['std']:.6f} "
            f"p50={metric['p50']:.6f} p95={metric['p95']:.6f}"
        )
    lines.append("")
    lines.append("By label")
    for label, payload in summary["by_label"].items():
        lines.append(f"- {label}: tokens={payload['token_count']:,}")
        for metric_name in ("H_norm", "logit_variance", "p_max"):
            metric = payload[metric_name]
            lines.append(
                f"    {metric_name}: mean={metric['mean']:.6f} std={metric['std']:.6f} "
                f"p50={metric['p50']:.6f} p95={metric['p95']:.6f}"
            )
    return "\n".join(lines) + "\n"


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect teacher-forced marginal stats on llm_eval rollout JSONLs and "
            "save a HuggingFace DatasetDict."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path or model id used to score the rollout text under teacher forcing.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Optional tokenizer override; defaults to --model_name_or_path.",
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
        "--output-dir",
        required=True,
        help="Output directory for the saved DatasetDict and summaries.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of rollout sequences per forward pass.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="Truncate tokenized prompt+response to at most this many tokens; use <=0 to disable.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on processed rollout sequences.",
    )
    parser.add_argument(
        "--stats-temperature",
        type=float,
        default=1.0,
        help="Temperature used to define q for entropy/p_max/logit-variance summaries.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cuda:0, cpu) used when --device-map is unset.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional device_map passed to from_pretrained().",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Token rows per in-memory chunk before flushing to Dataset.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress logging frequency in processed sequences; use <=0 to disable.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress messages to stderr.",
    )
    args = parser.parse_args()

    if args.stats_temperature <= 0:
        raise ValueError("--stats-temperature must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    os.makedirs(args.output_dir, exist_ok=True)

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
        paths = _resolve_inputs(group["spec"])
        resolved_groups.append({**group, "paths": paths})

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    dtype = _resolve_dtype(args.torch_dtype)
    model = _load_model(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    device = None
    if args.device_map is None:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)

    token_rows: List[Dict[str, Any]] = []
    token_datasets: List[Dataset] = []
    seq_rows: List[Dict[str, Any]] = []
    processed_sequences = 0
    processed_tokens = 0
    skipped_missing_prompt = 0
    skipped_missing_response = 0
    skipped_empty_response_tokens = 0
    truncated_sequences = 0
    seq_id_counter = 0

    token_features = Features(
        {
            "seq_id": Value("int64"),
            "response_t": Value("int32"),
            "token_id": Value("int32"),
            "H": Value("float32"),
            "H_norm": Value("float32"),
            "p_max": Value("float32"),
            "logit_variance": Value("float32"),
            "label": Value("string"),
            "dataset": Value("string"),
            "tag": Value("string"),
            "source_file": Value("string"),
            "source_path": Value("string"),
        }
    )
    seq_features = Features(
        {
            "seq_id": Value("int64"),
            "label": Value("string"),
            "dataset": Value("string"),
            "tag": Value("string"),
            "source_file": Value("string"),
            "source_path": Value("string"),
            "problem_index": Value("int32"),
            "sample_index": Value("int32"),
            "seed": Value("int32"),
            "score": Value("float32"),
            "prompt_len": Value("int32"),
            "response_len": Value("int32"),
            "response_offset": Value("int32"),
            "total_len": Value("int32"),
            "was_truncated": Value("bool"),
        }
    )

    def flush_tokens() -> None:
        if not token_rows:
            return
        token_datasets.append(Dataset.from_list(token_rows, features=token_features))
        token_rows.clear()

    def process_batch(batch_items: List[Dict[str, Any]]) -> None:
        nonlocal processed_tokens

        tensors = _pad_batch(batch_items, tokenizer.pad_token_id)
        input_ids_t = tensors["input_ids"]
        attention_mask_t = tensors["attention_mask"]
        label_mask_t = tensors["label_mask"]
        if device is not None:
            input_ids_t = input_ids_t.to(device)
            attention_mask_t = attention_mask_t.to(device)
            label_mask_t = label_mask_t.to(device)

        labels = input_ids_t.clone()
        labels[label_mask_t == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                output_hidden_states=False,
                use_cache=False,
            )

        logits = outputs.logits
        if logits is None:
            raise RuntimeError("Model did not return logits for rollout marginal collection.")

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return

        logits_valid = shift_logits[valid_mask]
        labels_valid = shift_labels[valid_mask]

        logits_f = logits_valid.float()
        scaled_logits = logits_f / float(args.stats_temperature)
        log_denom = torch.logsumexp(scaled_logits, dim=-1)
        log_probs = scaled_logits - log_denom.unsqueeze(-1)
        probs = torch.exp(log_probs)

        entropy = -(probs * log_probs).sum(-1)
        p_max = probs.max(dim=-1).values
        mean_z = (probs * logits_f).sum(-1)
        mean_z2 = (probs * logits_f.square()).sum(-1)
        logit_variance = torch.clamp_min(mean_z2 - mean_z.square(), 0.0)

        vocab = logits_f.size(-1)
        log_vocab = math.log(vocab) if vocab > 0 else 1.0
        h_norm = entropy / log_vocab

        response_pos = torch.cumsum(valid_mask.to(torch.long), dim=1)[valid_mask]

        finite_mask = (
            torch.isfinite(entropy)
            & torch.isfinite(h_norm)
            & torch.isfinite(p_max)
            & torch.isfinite(logit_variance)
        )
        mask_list = finite_mask.detach().cpu().tolist()

        entropy_list = entropy[finite_mask].detach().cpu().tolist()
        h_norm_list = h_norm[finite_mask].detach().cpu().tolist()
        p_max_list = p_max[finite_mask].detach().cpu().tolist()
        logit_variance_list = logit_variance[finite_mask].detach().cpu().tolist()
        token_id_list = labels_valid[finite_mask].detach().cpu().tolist()
        response_pos_list = response_pos[finite_mask].detach().cpu().tolist()

        valid_counts = valid_mask.sum(dim=1).detach().cpu().tolist()
        final_counts: List[int] = []
        offset = 0
        for count in valid_counts:
            if count == 0:
                final_counts.append(0)
                continue
            slice_mask = mask_list[offset : offset + count]
            final_counts.append(int(sum(slice_mask)))
            offset += count

        token_offset = 0
        for example, count in zip(batch_items, final_counts):
            for idx in range(count):
                token_rows.append(
                    {
                        "seq_id": int(example["seq_id"]),
                        "response_t": int(response_pos_list[token_offset + idx] + example["response_offset"]),
                        "token_id": int(token_id_list[token_offset + idx]),
                        "H": float(entropy_list[token_offset + idx]),
                        "H_norm": float(h_norm_list[token_offset + idx]),
                        "p_max": float(p_max_list[token_offset + idx]),
                        "logit_variance": float(logit_variance_list[token_offset + idx]),
                        "label": example["label"],
                        "dataset": example["dataset"],
                        "tag": example["tag"],
                        "source_file": example["source_file"],
                        "source_path": example["source_path"],
                    }
                )
            token_offset += count

        processed_tokens += len(entropy_list)
        if args.chunk_size > 0 and len(token_rows) >= args.chunk_size:
            flush_tokens()

    batch: List[Dict[str, Any]] = []

    for group in resolved_groups:
        _log(args.progress, f"[group] {group['label']} -> {len(group['paths'])} file(s)")
        for path in group["paths"]:
            _log(args.progress, f"[file] {path}")
            for row in _read_jsonl(path):
                if args.max_examples is not None and processed_sequences >= args.max_examples:
                    break

                prompt = _extract_scalar_text(row, ("prompt", "input"))
                if prompt is None:
                    skipped_missing_prompt += 1
                    continue

                response = _extract_scalar_text(
                    row,
                    ("response", "completion", "solution", "output", "generated", "answer"),
                )
                if response is None:
                    skipped_missing_response += 1
                    continue

                input_ids, label_mask, prompt_len, response_len, response_offset, was_truncated = (
                    _tokenize_prompt_response(
                        prompt,
                        response,
                        tokenizer,
                        args.max_seq_len if args.max_seq_len > 0 else None,
                    )
                )
                if response_len <= 0:
                    skipped_empty_response_tokens += 1
                    continue

                meta = _extract_metadata(row)
                dataset_name = _infer_dataset_name(group["dataset"], meta, path)
                tag_name = _infer_tag_name(group["tag"], path)

                seq_rows.append(
                    {
                        "seq_id": int(seq_id_counter),
                        "label": group["label"],
                        "dataset": dataset_name,
                        "tag": tag_name,
                        "source_file": path.name,
                        "source_path": str(path),
                        "problem_index": _as_int(meta.get("problem_index")),
                        "sample_index": _as_int(meta.get("sample_index")),
                        "seed": _as_int(meta.get("seed")),
                        "score": _as_float(meta.get("score")),
                        "prompt_len": int(prompt_len),
                        "response_len": int(response_len),
                        "response_offset": int(response_offset),
                        "total_len": int(len(input_ids)),
                        "was_truncated": bool(was_truncated),
                    }
                )
                batch.append(
                    {
                        "seq_id": int(seq_id_counter),
                        "input_ids": input_ids,
                        "label_mask": label_mask,
                        "response_offset": int(response_offset),
                        "label": group["label"],
                        "dataset": dataset_name,
                        "tag": tag_name,
                        "source_file": path.name,
                        "source_path": str(path),
                    }
                )
                seq_id_counter += 1
                processed_sequences += 1
                if was_truncated:
                    truncated_sequences += 1

                if len(batch) >= args.batch_size:
                    process_batch(batch)
                    batch.clear()

                if args.log_every > 0 and processed_sequences % args.log_every == 0:
                    _log(
                        args.progress,
                        f"[progress] sequences={processed_sequences:,} tokens={processed_tokens:,} "
                        f"truncated={truncated_sequences:,}",
                    )

            if args.max_examples is not None and processed_sequences >= args.max_examples:
                break
        if args.max_examples is not None and processed_sequences >= args.max_examples:
            break

    if batch:
        process_batch(batch)
        batch.clear()
    flush_tokens()

    if not token_datasets:
        raise RuntimeError("No token rows were collected from the provided rollout files.")

    tokens_dataset = concatenate_datasets(token_datasets) if len(token_datasets) > 1 else token_datasets[0]
    sequences_dataset = Dataset.from_list(seq_rows, features=seq_features)
    dataset_dict = DatasetDict({"tokens": tokens_dataset, "sequences": sequences_dataset})
    dataset_dict.save_to_disk(args.output_dir)

    summary = _summarize_token_dataset(tokens_dataset)
    summary_payload = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "stats_temperature": float(args.stats_temperature),
            "max_seq_len": int(args.max_seq_len),
            "batch_size": int(args.batch_size),
            "inputs": [group["spec"] for group in input_groups],
            "labels": [group["label"] for group in input_groups],
        },
        "counts": {
            "processed_sequences": int(processed_sequences),
            "processed_tokens": int(processed_tokens),
            "skipped_missing_prompt": int(skipped_missing_prompt),
            "skipped_missing_response": int(skipped_missing_response),
            "skipped_empty_response_tokens": int(skipped_empty_response_tokens),
            "truncated_sequences": int(truncated_sequences),
        },
        "summary": summary,
    }

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    summary_txt_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(_format_summary_text(summary))

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_sequences": int(processed_sequences),
                "processed_tokens": int(processed_tokens),
                "skipped_missing_prompt": int(skipped_missing_prompt),
                "skipped_missing_response": int(skipped_missing_response),
                "skipped_empty_response_tokens": int(skipped_empty_response_tokens),
                "truncated_sequences": int(truncated_sequences),
                "stats_temperature": float(args.stats_temperature),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[collect_rollout_marginals] Saved DatasetDict to {args.output_dir} | "
        f"sequences={processed_sequences:,} | tokens={processed_tokens:,}"
    )
    print(f"[collect_rollout_marginals] Summary: {summary_json_path}")


if __name__ == "__main__":
    main()
