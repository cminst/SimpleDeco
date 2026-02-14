"""
You can use this script to visualize the base LLM entropy vs token position
for one example.

Requires merged AutoDeco model, not just the small add-on head file!
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
            denom = window
        else:
            denom = idx + 1
        smoothed.append(running_sum / denom)
    return smoothed


def _value_to_hex(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
    # Blue (low) -> Red (high)
    hue = 210.0 * (1.0 - t)
    saturation = 0.85
    lightness = 0.75

    c = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
    h_prime = hue / 60.0
    x = c * (1.0 - abs(h_prime % 2.0 - 1.0))
    r1 = g1 = b1 = 0.0
    if 0.0 <= h_prime < 1.0:
        r1, g1, b1 = c, x, 0.0
    elif 1.0 <= h_prime < 2.0:
        r1, g1, b1 = x, c, 0.0
    elif 2.0 <= h_prime < 3.0:
        r1, g1, b1 = 0.0, c, x
    elif 3.0 <= h_prime < 4.0:
        r1, g1, b1 = 0.0, x, c
    elif 4.0 <= h_prime < 5.0:
        r1, g1, b1 = x, 0.0, c
    elif 5.0 <= h_prime <= 6.0:
        r1, g1, b1 = c, 0.0, x
    m = lightness - c / 2.0
    r = int(round((r1 + m) * 255))
    g = int(round((g1 + m) * 255))
    b = int(round((b1 + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def _display_token(token: str) -> str:
    return (
        token.replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


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
    parser.add_argument("--smooth_window", type=int, default=0)
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
    try:
        base_dtype = next(p for p in model.llm.parameters()).dtype
    except StopIteration:
        base_dtype = model.dtype
    if base_dtype is not None:
        for head_name in ("temp_head", "top_p_head"):
            head = getattr(model, head_name, None)
            if head is None:
                continue
            params = list(head.parameters())
            if params and params[0].dtype != base_dtype:
                head.to(dtype=base_dtype)

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

    logits = outputs.logits
    topk_indices = None
    topk_probs = None
    entropies = None
    if logits is not None:
        logits_f = logits.float()
        log_denom = torch.logsumexp(logits_f, dim=-1)
        log_probs = logits_f - log_denom.unsqueeze(-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(-1)
        entropies = entropy.squeeze(0).detach().cpu().tolist()
        if entropies and isinstance(entropies[0], list):
            entropies = entropies[0]

        topk_vals, topk_idx = torch.topk(logits_f, k=5, dim=-1)
        topk_prob = torch.exp(topk_vals - log_denom.unsqueeze(-1))
        topk_indices = topk_idx.squeeze(0).detach().cpu().tolist()
        topk_probs = topk_prob.squeeze(0).detach().cpu().tolist()
    if entropies is None:
        raise RuntimeError("Model did not return logits for entropy computation.")
    smoothed_entropies = _moving_average(entropies, args.smooth_window)
    token_ids = output_ids.tolist()

    # Align predictions to the token they generate (logits at position i -> token i+1).
    aligned_entropy: List[float | None] = [None] * len(token_ids)
    aligned_smoothed: List[float | None] = [None] * len(token_ids)
    aligned_topk_indices: List[List[int] | None] | None = (
        [None] * len(token_ids) if topk_indices is not None else None
    )
    aligned_topk_probs: List[List[float] | None] | None = (
        [None] * len(token_ids) if topk_probs is not None else None
    )
    max_src = min(len(entropies), len(token_ids) - 1)
    for idx in range(1, max_src + 1):
        aligned_entropy[idx] = entropies[idx - 1]
        if smoothed_entropies:
            aligned_smoothed[idx] = smoothed_entropies[idx - 1]
        if aligned_topk_indices is not None and aligned_topk_probs is not None:
            aligned_topk_indices[idx] = topk_indices[idx - 1]
            aligned_topk_probs[idx] = topk_probs[idx - 1]

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
    gen_start = min(prompt_len, len(token_ids))
    gen_entropies = [t for t in aligned_entropy[gen_start:] if t is not None]
    if gen_entropies:
        gen_min = min(gen_entropies)
        gen_max = max(gen_entropies)
    else:
        gen_min, gen_max = 0.0, 1.0
    token_spans_html: List[str] = []
    for idx in range(gen_start, len(token_ids)):
        token_text = decoded[spans[idx][0]:spans[idx][1]]
        escaped = html.escape(token_text)
        entropy_value = aligned_entropy[idx]
        if entropy_value is None:
            color = "#E0E0E0"
            tip_lines = ["entropy=n/a (no previous token)"]
        else:
            color = _value_to_hex(entropy_value, gen_min, gen_max)
            tip_lines = [f"entropy={entropy_value:.4f}"]
            if args.smooth_window > 1 and aligned_smoothed[idx] is not None:
                tip_lines.append(f"smooth={aligned_smoothed[idx]:.4f}")
            if aligned_topk_indices is not None and aligned_topk_probs is not None:
                topk_ids = aligned_topk_indices[idx]
                topk_ps = aligned_topk_probs[idx]
                if topk_ids is not None and topk_ps is not None:
                    tip_lines.append("top5:")
                    for tok_id, prob in zip(topk_ids, topk_ps):
                        tok = tokenizer.decode([tok_id], skip_special_tokens=False)
                        tok_disp = _display_token(tok)
                        tip_lines.append(f"{tok_disp}  {prob:.4f}")
        title = "&#10;".join(html.escape(line) for line in tip_lines)
        token_spans_html.append(
            f"<span class='tok' style='background-color: {color};' data-tip='{title}'>"
            f"{escaped}</span>"
        )
    generation_html = "".join(token_spans_html)

    output_text_path = os.path.join(args.output_dir, "entropy_trace.txt")
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(decoded)

    output_json_path = os.path.join(args.output_dir, "entropy_trace.jsonl")
    with open(output_json_path, "w", encoding="utf-8") as f:
        for idx, (token_id, (start, end)) in enumerate(zip(token_ids, spans)):
            entry = {
                "index": idx,
                "token_id": token_id,
                "token_text": decoded[start:end],
                "entropy": aligned_entropy[idx],
                "in_think": bool(think_mask[idx]),
                "in_code": bool(code_mask[idx]),
            }
            if args.smooth_window > 1:
                entry["entropy_smoothed"] = aligned_smoothed[idx]
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    fig, ax = plt.subplots(figsize=(14, 4))
    plot_entropy = [t if t is not None else float("nan") for t in aligned_entropy]
    ax.plot(range(len(plot_entropy)), plot_entropy, color="#2E3A59", linewidth=1.5, label="Base LLM entropy")
    if args.smooth_window > 1:
        plot_smoothed = [t if t is not None else float("nan") for t in aligned_smoothed]
        ax.plot(
            range(len(plot_smoothed)),
            plot_smoothed,
            color="#E07A5F",
            linewidth=1.8,
            label=f"Smoothed (window={args.smooth_window})",
        )

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
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy vs Token Position")
    ax.set_xlim(0, len(token_ids) - 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    output_png_path = os.path.join(args.output_dir, "entropy_trace.png")
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=160)

    output_html_path = os.path.join(args.output_dir, "entropy_trace.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        smoothing_note = ""
        if args.smooth_window > 1:
            smoothing_note = f" Smoothed with window={args.smooth_window}."
        f.write(
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>Entropy Trace</title>\n"
            "<style>\n"
            "body { font-family: Arial, sans-serif; }\n"
            ".token-box { white-space: pre-wrap; word-break: break-word; "
            "border: 1px solid #DDD; padding: 12px; border-radius: 8px; "
            "background: #FAFAFA; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; }\n"
            ".tok { padding: 0 1px; border-radius: 3px; position: relative; }\n"
            ".tok:hover::after { "
            "content: attr(data-tip); "
            "position: absolute; "
            "left: 0; top: 1.2em; "
            "background: #1F2937; color: #F9FAFB; "
            "padding: 8px 10px; border-radius: 8px; "
            "font-size: 12px; line-height: 1.3; "
            "white-space: pre; "
            "box-shadow: 0 6px 18px rgba(0,0,0,0.25); "
            "z-index: 5; min-width: 200px; }\n"
            ".legend { font-size: 12px; color: #444; }\n"
            "</style></head>\n"
            "<body>\n"
            f"<h2>Entropy Trace ({args.dataset_name}:{split_name} idx={args.row_index})</h2>\n"
            "<p>Blue shading: &lt;think&gt; spans. Yellow shading: code blocks."
            " Entropy values are aligned to the token they generate (first token has no prediction)."
            f"{smoothing_note}</p>\n"
            f"<img src='entropy_trace.png' style='max-width: 100%; height: auto;' />\n"
            f"<p>Prompt length: {prompt_len} tokens, total length: {len(token_ids)} tokens.</p>\n"
            "<h3>Generated Tokens (colored by base LLM entropy)</h3>\n"
            f"<div class='legend'>Min: {gen_min:.4f} &nbsp; Max: {gen_max:.4f} "
            f"&nbsp; (hover a token for exact value)</div>\n"
            f"<div class='token-box'>{generation_html}</div>\n"
            "</body></html>\n"
        )


if __name__ == "__main__":
    main()
