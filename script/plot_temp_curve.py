#!/usr/bin/env python3
"""
Plot predicted temperature vs token position for a single example.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Tuple

import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from model.templlm_auto import AutoDecoModelForCausalLM


def _resolve_local_dataset_file(dataset_name: str) -> str | None:
    candidates = [dataset_name]
    if not os.path.isabs(dataset_name):
        candidates.append(os.path.join("data", dataset_name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _pick_text_field(columns: List[str], preferred: str | None) -> str:
    if preferred and preferred in columns:
        return preferred
    if "messages" in columns:
        return "messages"
    if "conversations" in columns:
        return "conversations"
    if "prompt" in columns:
        return "prompt"
    return columns[0]


def _load_dataset_split(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str | None,
) -> Tuple[Any, str]:
    local_file = _resolve_local_dataset_file(dataset_name)
    if local_file is not None:
        dataset = load_dataset("json", data_files=local_file)
    else:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)

    if not hasattr(dataset, "keys"):
        dataset = {"train": dataset}

    splits = list(dataset.keys())
    if not splits:
        raise ValueError("Loaded dataset has no splits.")

    split_name = dataset_split if dataset_split in splits else splits[0]
    return dataset[split_name], split_name


def _strip_trailing_assistant(convo: List[dict[str, Any]]) -> List[dict[str, Any]]:
    trimmed = list(convo)
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed = trimmed[:-1]
    return trimmed


def _build_prompt_from_row(
    row: dict[str, Any],
    text_field: str,
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool,
    enable_thinking: bool,
    strip_assistant: bool,
    user_suffix: str | None,
) -> str:
    value = row.get(text_field)
    if value is None:
        raise ValueError(f"Row does not contain field '{text_field}'.")

    if text_field in {"messages", "conversations"}:
        if not isinstance(value, list):
            raise ValueError(f"Expected list for '{text_field}', got {type(value).__name__}.")
        convo = value
        if strip_assistant:
            convo = _strip_trailing_assistant(convo)
        if user_suffix:
            for turn in reversed(convo):
                if turn.get("role") == "user":
                    turn["content"] = f"{turn.get('content','')}{user_suffix}"
                    break
        try:
            return tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    if not isinstance(value, str):
        value = str(value)
    if user_suffix:
        value = f"{value}{user_suffix}"
    return value


def _find_ranges(text: str, start_tag: str, end_tag: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(start_tag, pos)
        if start == -1:
            break
        content_start = start + len(start_tag)
        end = text.find(end_tag, content_start)
        if end == -1:
            ranges.append((content_start, len(text)))
            break
        ranges.append((content_start, end))
        pos = end + len(end_tag)
    return ranges


def _mask_from_ranges(spans: List[Tuple[int, int]], ranges: List[Tuple[int, int]]) -> List[bool]:
    mask = []
    for start, end in spans:
        inside = False
        for r_start, r_end in ranges:
            if start < r_end and end > r_start:
                inside = True
                break
        mask.append(inside)
    return mask


def _mask_to_segments(mask: List[bool]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--dataset_text_field", default=None)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--add_generation_prompt", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--strip_assistant", action="store_true")
    parser.add_argument("--user_suffix", default=None)
    parser.add_argument("--output_dir", default="figure")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_split, split_name = _load_dataset_split(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    text_field = _pick_text_field(dataset_split.column_names, args.dataset_text_field)
    row = dataset_split[int(args.row_index)]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    prompt = _build_prompt_from_row(
        row=row,
        text_field=text_field,
        tokenizer=tokenizer,
        add_generation_prompt=args.add_generation_prompt,
        enable_thinking=args.enable_thinking,
        strip_assistant=args.strip_assistant,
        user_suffix=args.user_suffix,
    )

    dtype = getattr(torch, args.torch_dtype, torch.bfloat16)
    model = AutoDecoModelForCausalLM.from_pretrained(
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

    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
        output_ids = output_ids[0]
        prompt_len = inputs["input_ids"].shape[-1]

        attention_mask = torch.ones_like(output_ids).unsqueeze(0)
        forward_ids = output_ids.unsqueeze(0)
        outputs = model(
            input_ids=forward_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

    temps = outputs.temp_logits.squeeze(-1).detach().cpu().tolist()
    token_ids = output_ids.tolist()

    decoded = ""
    spans: List[Tuple[int, int]] = []
    for token_id in token_ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        start = len(decoded)
        decoded += token_text
        spans.append((start, len(decoded)))

    think_ranges = _find_ranges(decoded, "<think>", "</think>")
    code_ranges = _find_ranges(decoded, "```", "```")
    think_mask = _mask_from_ranges(spans, think_ranges)
    code_mask = _mask_from_ranges(spans, code_ranges)

    output_text_path = os.path.join(args.output_dir, "temp_trace.txt")
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(decoded)

    output_json_path = os.path.join(args.output_dir, "temp_trace.jsonl")
    with open(output_json_path, "w", encoding="utf-8") as f:
        for idx, (token_id, temp, (start, end)) in enumerate(zip(token_ids, temps, spans)):
            f.write(json.dumps({
                "index": idx,
                "token_id": token_id,
                "token_text": decoded[start:end],
                "temperature": float(temp),
                "in_think": bool(think_mask[idx]),
                "in_code": bool(code_mask[idx]),
            }, ensure_ascii=False) + "\n")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(len(temps)), temps, color="#2E3A59", linewidth=1.5, label="Predicted temperature")

    think_segments = _mask_to_segments(think_mask)
    code_segments = _mask_to_segments(code_mask)
    think_labeled = False
    code_labeled = False
    for start, end in think_segments:
        ax.axvspan(start, end, color="#CFE8FF", alpha=0.35, label="Inside <think>" if not think_labeled else None)
        think_labeled = True
    for start, end in code_segments:
        ax.axvspan(start, end, color="#FFF2B2", alpha=0.35, label="Inside code" if not code_labeled else None)
        code_labeled = True

    ax.axvline(prompt_len - 1, color="#888888", linestyle="--", linewidth=1, label="Prompt end")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Predicted temperature")
    ax.set_title("Temperature vs Token Position")
    ax.set_xlim(0, len(temps) - 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    output_png_path = os.path.join(args.output_dir, "temp_trace.png")
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=160)

    output_html_path = os.path.join(args.output_dir, "temp_trace.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>Temp Trace</title></head>\n"
            "<body style='font-family: Arial, sans-serif;'>\n"
            f"<h2>Temperature Trace ({args.dataset_name}:{split_name} idx={args.row_index})</h2>\n"
            "<p>Blue shading: &lt;think&gt; spans. Yellow shading: code blocks.</p>\n"
            f"<img src='temp_trace.png' style='max-width: 100%; height: auto;' />\n"
            f"<p>Prompt length: {prompt_len} tokens, total length: {len(temps)} tokens.</p>\n"
            "</body></html>\n"
        )


if __name__ == "__main__":
    main()
