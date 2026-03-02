"""
You can use this script to visualize the base LLM entropy vs token position
for one example.

Requires merged AutoDeco model, not just the small add-on head file!
"""
from __future__ import annotations

import argparse
import bisect
import html
import json
import math
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

TEMP_COLOR_MIN = 0.0
TEMP_COLOR_MAX = 1.5


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


def _pearsonr(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = 0.0
    var_y = 0.0
    cov = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        var_x += dx * dx
        var_y += dy * dy
        cov += dx * dy
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


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
    parser.add_argument("--comparison_top_k", type=int, default=12)
    parser.add_argument("--comparison_context_lines", type=int, default=1)
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
    temps = None
    temp_entropies = None
    temp_logits = getattr(outputs, "temp_logits", None)
    if temp_logits is not None:
        temps_tensor = temp_logits.squeeze(-1).float()
        temps = temps_tensor.detach().cpu().tolist()
        if temps and isinstance(temps[0], list):
            temps = temps[0]
        if logits is not None:
            if temps_tensor.dim() == 2:
                temps_tensor = temps_tensor.squeeze(0)
            safe_temps = torch.where(
                temps_tensor > 0.0,
                temps_tensor,
                torch.full_like(temps_tensor, float("nan")),
            )
            scaled_logits = logits_f / safe_temps.unsqueeze(-1)
            log_denom_temp = torch.logsumexp(scaled_logits, dim=-1)
            log_probs_temp = scaled_logits - log_denom_temp.unsqueeze(-1)
            probs_temp = torch.exp(log_probs_temp)
            entropy_temp = -(probs_temp * log_probs_temp).sum(-1)
            temp_entropies = entropy_temp.squeeze(0).detach().cpu().tolist()
            if temp_entropies and isinstance(temp_entropies[0], list):
                temp_entropies = temp_entropies[0]
    smoothed_entropies = _moving_average(entropies, args.smooth_window)
    token_ids = output_ids.tolist()

    # Align predictions to the token they generate (logits at position i -> token i+1).
    aligned_entropy: List[float | None] = [None] * len(token_ids)
    aligned_smoothed: List[float | None] = [None] * len(token_ids)
    aligned_temps: List[float | None] | None = [None] * len(token_ids) if temps is not None else None
    aligned_temp_entropy: List[float | None] | None = (
        [None] * len(token_ids) if temp_entropies is not None else None
    )
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
        if aligned_temps is not None and temps is not None and idx - 1 < len(temps):
            aligned_temps[idx] = temps[idx - 1]
        if aligned_temp_entropy is not None and temp_entropies is not None and idx - 1 < len(temp_entropies):
            val = temp_entropies[idx - 1]
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                aligned_temp_entropy[idx] = None
            else:
                aligned_temp_entropy[idx] = float(val)
        if aligned_topk_indices is not None and aligned_topk_probs is not None:
            if topk_indices is not None and topk_probs is not None and idx - 1 < len(topk_indices):
                aligned_topk_indices[idx] = topk_indices[idx - 1]
                aligned_topk_probs[idx] = topk_probs[idx - 1]

    delta_by_idx: List[float | None] = [None] * len(token_ids)
    if aligned_temp_entropy is not None:
        for idx in range(len(token_ids)):
            t = aligned_temp_entropy[idx]
            e = aligned_entropy[idx]
            if t is None or e is None:
                continue
            delta_by_idx[idx] = abs(t - e)

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
    gen_temp_min, gen_temp_max = TEMP_COLOR_MIN, TEMP_COLOR_MAX
    gen_deltas: List[float] = []
    if aligned_temp_entropy is not None:
        for idx in range(gen_start, len(token_ids)):
            t = aligned_temp_entropy[idx]
            e = aligned_entropy[idx]
            if t is None or e is None:
                continue
            gen_deltas.append(abs(t - e))
    if gen_deltas:
        gen_delta_min = min(gen_deltas)
        gen_delta_max = max(gen_deltas)
    else:
        gen_delta_min, gen_delta_max = 0.0, 1.0
    token_spans_html: List[str] = []
    for idx in range(gen_start, len(token_ids)):
        token_text = decoded[spans[idx][0]:spans[idx][1]]
        escaped = html.escape(token_text)
        entropy_value = aligned_entropy[idx]
        if entropy_value is None:
            entropy_color = "#E0E0E0"
            tip_lines = ["entropy=n/a (no previous token)"]
        else:
            entropy_color = _value_to_hex(entropy_value, gen_min, gen_max)
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
                    tip_lines.append(f"'{tok_disp}'  {prob:.4f}")
        temp_value = aligned_temps[idx] if aligned_temps is not None else None
        if temp_value is None:
            temp_color = "#E0E0E0"
            if aligned_temps is not None:
                tip_lines.append("temp=n/a")
        else:
            temp_color = _value_to_hex(temp_value, gen_temp_min, gen_temp_max)
            tip_lines.append(f"temp={temp_value:.4f}")
        temp_entropy_value = aligned_temp_entropy[idx] if aligned_temp_entropy is not None else None
        if temp_entropy_value is not None:
            tip_lines.append(f"entropy@temp={temp_entropy_value:.4f}")
        delta_value = None
        if temp_entropy_value is not None and entropy_value is not None:
            delta_value = abs(temp_entropy_value - entropy_value)
            tip_lines.append(f"delta=|entropy@temp-entropy|={delta_value:.4f}")
            delta_color = _value_to_hex(delta_value, gen_delta_min, gen_delta_max)
        else:
            delta_color = "#E0E0E0"
        title = "&#10;".join(html.escape(line) for line in tip_lines)
        entropy_attr = "" if entropy_value is None else f"{entropy_value:.6f}"
        temp_attr = "" if temp_value is None else f"{temp_value:.6f}"
        delta_attr = "" if delta_value is None else f"{delta_value:.6f}"
        token_spans_html.append(
            "<span class='tok' "
            f"style='background-color: {entropy_color};' "
            f"data-tip='{title}' "
            f"data-entropy='{entropy_attr}' "
            f"data-temp='{temp_attr}' "
            f"data-delta='{delta_attr}' "
            f"data-color-entropy='{entropy_color}' "
            f"data-color-temp='{temp_color}' "
            f"data-color-delta='{delta_color}'>"
            f"{escaped}</span>"
        )
    generation_html = "".join(token_spans_html)

    pearson_r = None
    pearson_n = 0
    if aligned_temps is not None:
        xs: List[float] = []
        ys: List[float] = []
        for idx in range(gen_start, len(token_ids)):
            t = aligned_temps[idx]
            e = aligned_entropy[idx]
            if t is None or e is None:
                continue
            xs.append(t)
            ys.append(e)
        pearson_n = len(xs)
        pearson_r = _pearsonr(xs, ys)

    if aligned_temps is None:
        print("pearsonr (temp vs entropy, generated tokens): n/a (no temp head)")
    elif pearson_r is None:
        print(f"pearsonr (temp vs entropy, generated tokens): n={pearson_n} (insufficient/zero variance)")
    else:
        print(f"pearsonr (temp vs entropy, generated tokens): r={pearson_r:.4f}, n={pearson_n}")

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

    output_comparison_path = os.path.join(args.output_dir, "entropy_trace_comparison.txt")
    prompt_char_end = spans[gen_start][0] if gen_start < len(spans) else len(decoded)
    prompt_text = decoded[:prompt_char_end]
    generated_text = decoded[prompt_char_end:]
    selected_indices: List[int] = []
    if args.comparison_top_k > 0 and aligned_temp_entropy is not None:
        candidates: List[Tuple[float, int]] = []
        for idx in range(gen_start, len(token_ids)):
            delta_val = delta_by_idx[idx]
            if delta_val is None:
                continue
            candidates.append((delta_val, idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in candidates[: args.comparison_top_k]]
    selected_set = set(selected_indices)
    if not selected_indices:
        notice = ""
        if aligned_temp_entropy is None:
            notice = "\n[delta comparison unavailable: no temp head]\n"
        with open(output_comparison_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
            if not prompt_text.endswith("\n"):
                f.write("\n")
            if notice:
                f.write(notice)
            f.write(generated_text)
    else:
        context_lines = max(0, args.comparison_context_lines)
        base_lines = generated_text.splitlines()
        keep_lines: set[int] = set()
        for idx in selected_indices:
            token_start_rel = spans[idx][0] - prompt_char_end
            line_idx = generated_text.count("\n", 0, token_start_rel)
            for line_id in range(line_idx - context_lines, line_idx + context_lines + 1):
                if 0 <= line_id < len(base_lines):
                    keep_lines.add(line_id)

        decorated_chunks: List[str] = []
        for idx in range(gen_start, len(token_ids)):
            token_text = decoded[spans[idx][0]:spans[idx][1]]
            if idx in selected_set:
                token_disp = _display_token(token_text).replace("'", "\\'")
                parts = [f"picked token: '{token_disp}'"]
                entropy_val = aligned_entropy[idx]
                temp_val = aligned_temps[idx] if aligned_temps is not None else None
                temp_entropy_val = aligned_temp_entropy[idx] if aligned_temp_entropy is not None else None
                delta_val = delta_by_idx[idx]
                if entropy_val is not None:
                    parts.append(f"entropy: {entropy_val:.4f}")
                if temp_entropy_val is not None:
                    parts.append(f"entropy@temp: {temp_entropy_val:.4f}")
                if delta_val is not None:
                    parts.append(f"delta: {delta_val:.4f}")
                if temp_val is not None:
                    parts.append(f"temp: {temp_val:.4f}")
                if aligned_topk_indices is not None and aligned_topk_probs is not None:
                    topk_ids = aligned_topk_indices[idx]
                    topk_ps = aligned_topk_probs[idx]
                if topk_ids is not None and topk_ps is not None:
                    top5_items = []
                    for tok_id, prob in zip(topk_ids, topk_ps):
                        tok = tokenizer.decode([tok_id], skip_special_tokens=False)
                        tok_disp = _display_token(tok).replace("'", "\\'")
                        top5_items.append(f"'{tok_disp}' ({prob:.4f})")
                    if top5_items:
                        parts.append(f"top5: {', '.join(top5_items)}")
                annotation = "{{{" + " | ".join(parts) + "}}}"
                decorated_chunks.append(annotation)
            decorated_chunks.append(token_text)

        decorated_text = "".join(decorated_chunks)
        decorated_lines = decorated_text.splitlines()
        summary_lines: List[str] = []
        skipping = False
        for idx, line in enumerate(decorated_lines):
            if idx in keep_lines:
                summary_lines.append(line)
                skipping = False
            else:
                if not skipping:
                    summary_lines.append("...")
                    skipping = True
        summary_text = "\n".join(summary_lines)
        if decorated_text.endswith("\n"):
            summary_text += "\n"
        with open(output_comparison_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
            if not prompt_text.endswith("\n"):
                f.write("\n")
            f.write(summary_text)

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
        has_temp = aligned_temps is not None
        pearson_display = (
            "n/a (no temp head)"
            if aligned_temps is None
            else ("n/a" if pearson_r is None else f"{pearson_r:.4f}")
        )
        pearson_suffix = (
            ""
            if aligned_temps is None
            else (f"(n={pearson_n})" if pearson_n else "")
        )
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
            ".controls { display: flex; align-items: center; gap: 8px; margin: 8px 0 6px; }\n"
            ".controls label { font-size: 13px; color: #333; }\n"
            "</style></head>\n"
            "<body>\n"
            f"<h2>Entropy Trace ({args.dataset_name}:{split_name} idx={args.row_index})</h2>\n"
            f"<p>Pearson r (temp vs entropy, generated tokens): {pearson_display} {pearson_suffix}</p>\n"
            "<p>Blue shading: &lt;think&gt; spans. Yellow shading: code blocks."
            " Entropy values are aligned to the token they generate (first token has no prediction)."
            " Delta uses entropy after applying the predicted temperature to logits."
            f"{smoothing_note}</p>\n"
            f"<img src='entropy_trace.png' style='max-width: 100%; height: auto;' />\n"
            f"<p>Prompt length: {prompt_len} tokens, total length: {len(token_ids)} tokens.</p>\n"
            "<h3>Generated Tokens</h3>\n"
            "<div class='controls'>\n"
            "<label for='color-mode'>Color by</label>\n"
            "<select id='color-mode'>\n"
            "<option value='entropy' selected>Base LLM entropy</option>\n"
            f"<option value='temp' {'disabled' if not has_temp else ''}>SimpleDeco predicted temperature</option>\n"
            f"<option value='delta' {'disabled' if not has_temp else ''}>Delta (|entropy@temp - entropy|)</option>\n"
            "</select>\n"
            "</div>\n"
            f"<div class='legend'>Min: <span id='legend-min'>{gen_min:.4f}</span> &nbsp; "
            f"Max: <span id='legend-max'>{gen_max:.4f}</span> &nbsp; "
            "(hover a token for exact value)</div>\n"
            f"<div class='token-box'>{generation_html}</div>\n"
            "<script>\n"
            "const modeSelect = document.getElementById('color-mode');\n"
            "const legendMin = document.getElementById('legend-min');\n"
            "const legendMax = document.getElementById('legend-max');\n"
            "const tokens = Array.from(document.querySelectorAll('.tok'));\n"
            f"const ranges = {{\n"
            f"  entropy: {{min: {gen_min:.6f}, max: {gen_max:.6f}}},\n"
            f"  temp: {{min: {gen_temp_min:.6f}, max: {gen_temp_max:.6f}}},\n"
            f"  delta: {{min: {gen_delta_min:.6f}, max: {gen_delta_max:.6f}}},\n"
            "};\n"
            "function applyMode(mode) {\n"
            "  tokens.forEach((tok) => {\n"
            "    const key = `color${mode[0].toUpperCase()}${mode.slice(1)}`;\n"
            "    const color = tok.dataset[key] || '#E0E0E0';\n"
            "    tok.style.backgroundColor = color;\n"
            "  });\n"
            "  if (ranges[mode]) {\n"
            "    legendMin.textContent = ranges[mode].min.toFixed(4);\n"
            "    legendMax.textContent = ranges[mode].max.toFixed(4);\n"
            "  }\n"
            "}\n"
            "modeSelect.addEventListener('change', (event) => applyMode(event.target.value));\n"
            "applyMode(modeSelect.value);\n"
            "</script>\n"
            "</body></html>\n"
        )


if __name__ == "__main__":
    main()
