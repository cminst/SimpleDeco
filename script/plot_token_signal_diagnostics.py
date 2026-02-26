"""
Token-level diagnostics for SimpleDeco heads using teacher forcing.

Computes:
  - AUROC for T_pred, entropy, and -P_max on incorrect tokens.
  - Conditional AUROC for T_pred on confident tokens (P_max > threshold or top quantile).
  - Scatter plot of entropy vs T_pred colored by correctness.

Requires merged AutoDeco model, not just the small add-on head file!
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
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
    for name in ("messages", "conversations", "text", "prompt"):
        if name in columns:
            return name
    return columns[0]


def _load_dataset_split(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str | None,
):
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


def _looks_like_chat_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(turn, dict)
            and isinstance(turn.get("role"), str)
            and isinstance(turn.get("content"), str)
            for turn in value
        )
    )


def _strip_trailing_assistant(convo: List[dict[str, Any]]) -> List[dict[str, Any]]:
    trimmed = list(convo)
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed = trimmed[:-1]
    return trimmed


def _apply_chat_template(
    tokenizer: AutoTokenizer,
    convo: List[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    return_assistant_mask: bool,
    tools: Any | None,
    extra_kwargs: Dict[str, Any] | None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if return_assistant_mask:
        kwargs["return_assistant_tokens_mask"] = True
    try:
        return tokenizer.apply_chat_template(convo, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(convo, **kwargs)


def _tokenize_prompt_completion(
    row: dict[str, Any],
    tokenizer: AutoTokenizer,
    assistant_only: bool,
    completion_only: bool,
    enable_thinking: bool,
) -> Tuple[List[int], List[int] | None, bool]:
    prompt_value = row.get("prompt")
    completion_value = row.get("completion")

    output_ids: List[int]
    label_mask: List[int] | None = None
    assistant_mask_used = False

    if _looks_like_chat_messages(prompt_value) and _looks_like_chat_messages(completion_value):
        prompt_ids = tokenizer.apply_chat_template(
            prompt_value,
            tokenize=True,
            tools=row.get("tools"),
            **row.get("chat_template_kwargs", {}),
        )
        prompt_completion = _apply_chat_template(
            tokenizer,
            prompt_value + completion_value,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_assistant_mask=assistant_only,
            tools=row.get("tools"),
            extra_kwargs=row.get("chat_template_kwargs"),
        )
        output_ids = prompt_completion["input_ids"]
        completion_mask = [0] * len(prompt_ids) + [1] * (len(output_ids) - len(prompt_ids))
        if assistant_only:
            assistant_masks = prompt_completion.get("assistant_masks")
            if assistant_masks is not None and len(assistant_masks) == len(output_ids):
                label_mask = [int(x) for x in assistant_masks]
                assistant_mask_used = True
        if label_mask is None and completion_only:
            label_mask = completion_mask
        return output_ids, label_mask, assistant_mask_used

    prompt_text = prompt_value if isinstance(prompt_value, str) else str(prompt_value)
    completion_text = completion_value if isinstance(completion_value, str) else str(completion_value)
    prompt_ids = tokenizer(text=prompt_text)["input_ids"]
    output_ids = tokenizer(text=prompt_text + completion_text)["input_ids"]
    if tokenizer.bos_token_id is not None:
        if prompt_ids and prompt_ids[0] == tokenizer.bos_token_id:
            prompt_ids = prompt_ids[1:]
        if output_ids and output_ids[0] == tokenizer.bos_token_id:
            output_ids = output_ids[1:]
    completion_mask = [0] * len(prompt_ids) + [1] * (len(output_ids) - len(prompt_ids))
    if completion_only:
        label_mask = completion_mask
    return output_ids, label_mask, assistant_mask_used


def _tokenize_row(
    row: dict[str, Any],
    text_field: str,
    tokenizer: AutoTokenizer,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    strip_assistant: bool,
    user_suffix: str | None,
    assistant_only: bool,
    completion_only: bool,
) -> Tuple[List[int], List[int] | None, bool]:
    if row.get("prompt") is not None and row.get("completion") is not None:
        return _tokenize_prompt_completion(
            row,
            tokenizer,
            assistant_only=assistant_only,
            completion_only=completion_only,
            enable_thinking=enable_thinking,
        )

    conversation = None
    for key in (text_field, "messages", "conversations"):
        if key in row and _looks_like_chat_messages(row[key]):
            conversation = row[key]
            break
    if conversation is not None:
        convo = list(conversation)
        if strip_assistant:
            convo = _strip_trailing_assistant(convo)
        if user_suffix:
            for turn in reversed(convo):
                if turn.get("role") == "user":
                    turn["content"] = f"{turn.get('content','')}{user_suffix}"
                    break
        processed = _apply_chat_template(
            tokenizer,
            convo,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_assistant_mask=assistant_only,
            tools=row.get("tools"),
            extra_kwargs=row.get("chat_template_kwargs"),
        )
        input_ids = processed["input_ids"]
        if assistant_only:
            assistant_masks = processed.get("assistant_masks")
            if assistant_masks is not None and len(assistant_masks) == len(input_ids):
                return input_ids, [int(x) for x in assistant_masks], True
        return input_ids, None, False

    value = row.get(text_field)
    if value is None:
        raise KeyError(f"Row does not contain field '{text_field}'.")
    if not isinstance(value, str):
        value = str(value)
    if user_suffix:
        value = f"{value}{user_suffix}"
    input_ids = tokenizer(text=value)["input_ids"]
    return input_ids, None, False


def _pad_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    max_len = max(len(example["input_ids"]) for example in batch)
    input_ids = []
    attention_mask = []
    label_mask = []
    for example in batch:
        ids = example["input_ids"]
        mask = example["label_mask"]
        padding = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        if mask is None:
            label_mask.append([1] * len(ids) + [0] * padding)
        else:
            label_mask.append(mask + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label_mask": torch.tensor(label_mask, dtype=torch.long),
    }


def _roc_auc_score_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score as _sk_roc_auc
    except Exception:
        return _roc_auc_score_fallback(y_true, y_score)
    try:
        return float(_sk_roc_auc(y_true, y_score))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run token-level signal diagnostics for a merged AutoDeco model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path or Hugging Face model id for the merged AutoDeco/SimpleDeco checkpoint.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name, local JSON file path, or dataset file name under data/.",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="Optional dataset config name passed to datasets.load_dataset().",
    )
    parser.add_argument(
        "--dataset_split",
        default=None,
        help="Dataset split to use; if omitted, the first available split is used.",
    )
    parser.add_argument(
        "--dataset_text_field",
        default=None,
        help="Text/chat field name; if omitted, the script auto-detects one.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of dataset examples to process.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Stop once this many valid tokens have been collected.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of examples processed together per forward pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and scatter subsampling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset indices before applying max_examples.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g., cuda:0, cpu) used when --device_map is unset.",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Device map passed to from_pretrained (e.g., auto).",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        help="Torch dtype name used when loading the model.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom modeling code from remote repositories.",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Add generation prompt when applying the chat template.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable the tokenizer's thinking mode when supported.",
    )
    parser.add_argument(
        "--strip_assistant",
        action="store_true",
        help="Strip trailing assistant turns before applying chat templates.",
    )
    parser.add_argument(
        "--user_suffix",
        default=None,
        help="Optional suffix appended to the final user turn or plain text.",
    )
    parser.add_argument(
        "--assistant_only",
        action="store_true",
        help="Score only assistant tokens when assistant masks are available.",
    )
    parser.add_argument(
        "--completion_only",
        action="store_true",
        help="For prompt/completion rows, score completion tokens only.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="P_max threshold for selecting confident tokens.",
    )
    parser.add_argument(
        "--confidence_quantile",
        type=float,
        default=None,
        help="If set, use this p_max quantile (0, 1) instead of a fixed threshold.",
    )
    parser.add_argument(
        "--min_confident_tokens",
        type=int,
        default=50,
        help="Minimum confident tokens required before confident-only AUROC is computed.",
    )
    parser.add_argument(
        "--max_scatter_points",
        type=int,
        default=20000,
        help="Maximum points to draw in the entropy vs T_pred scatter plot.",
    )
    parser.add_argument(
        "--output_dir",
        default="figure/token_signal_diagnostics",
        help="Output directory for summaries and generated plots.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Progress logging frequency in processed examples; use <=0 to disable.",
    )
    args = parser.parse_args()

    if args.confidence_quantile is not None:
        if not 0.0 < args.confidence_quantile < 1.0:
            raise ValueError("--confidence_quantile must be in (0, 1).")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_split, split_name = _load_dataset_split(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    text_field = _pick_text_field(dataset_split.column_names, args.dataset_text_field)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

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

    total_examples = len(dataset_split)
    indices = list(range(total_examples))
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(indices)
    if args.max_examples is not None:
        indices = indices[: args.max_examples]

    t_pred_values: List[float] = []
    entropy_values: List[float] = []
    p_max_values: List[float] = []
    incorrect_values: List[int] = []
    assistant_mask_used_examples = 0
    assistant_mask_missing_examples = 0

    processed_examples = 0
    processed_tokens = 0
    used_temp_head = False

    batch: List[Dict[str, Any]] = []
    for idx in indices:
        row = dataset_split[int(idx)]
        input_ids, label_mask, assistant_mask_used = _tokenize_row(
            row=row,
            text_field=text_field,
            tokenizer=tokenizer,
            add_generation_prompt=args.add_generation_prompt,
            enable_thinking=args.enable_thinking,
            strip_assistant=args.strip_assistant,
            user_suffix=args.user_suffix,
            assistant_only=args.assistant_only,
            completion_only=args.completion_only,
        )
        if assistant_mask_used:
            assistant_mask_used_examples += 1
        if args.assistant_only and not assistant_mask_used:
            assistant_mask_missing_examples += 1
        if not input_ids or len(input_ids) < 2:
            continue
        batch.append({"input_ids": input_ids, "label_mask": label_mask})
        if len(batch) < args.batch_size:
            continue

        tensors = _pad_batch(batch, tokenizer.pad_token_id)
        batch = []

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
            raise RuntimeError("Model did not return logits for entropy computation.")

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            processed_examples += attention_mask_t.size(0)
            continue

        logits_valid = shift_logits[valid_mask]
        labels_valid = shift_labels[valid_mask]

        logits_f = logits_valid.float()
        log_denom = torch.logsumexp(logits_f, dim=-1)
        log_probs = logits_f - log_denom.unsqueeze(-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(-1)
        p_max = torch.exp(logits_f.max(dim=-1).values - log_denom)
        pred_ids = logits_valid.argmax(dim=-1)
        correct = pred_ids == labels_valid

        temp_logits = getattr(outputs, "temp_logits", None)
        temp_valid = None
        if temp_logits is not None:
            temp_valid = temp_logits[:, :-1, :].squeeze(-1)[valid_mask]
            used_temp_head = True

        finite_mask = torch.isfinite(entropy) & torch.isfinite(p_max)
        if temp_valid is not None:
            finite_mask &= torch.isfinite(temp_valid)

        if finite_mask.any():
            entropy_values.extend(entropy[finite_mask].detach().cpu().tolist())
            p_max_values.extend(p_max[finite_mask].detach().cpu().tolist())
            incorrect_values.extend((~correct[finite_mask]).to(torch.int).detach().cpu().tolist())
            if temp_valid is not None:
                t_pred_values.extend(temp_valid[finite_mask].detach().cpu().tolist())

        processed_tokens += int(finite_mask.sum().item())
        processed_examples += attention_mask_t.size(0)

        if args.max_tokens is not None and processed_tokens >= args.max_tokens:
            break
        if args.log_every > 0 and processed_examples % args.log_every == 0:
            print(f"[{processed_examples}/{len(indices)}] tokens={processed_tokens}")

    if batch and (args.max_tokens is None or processed_tokens < args.max_tokens):
        tensors = _pad_batch(batch, tokenizer.pad_token_id)
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
            raise RuntimeError("Model did not return logits for entropy computation.")
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        if valid_mask.any():
            logits_valid = shift_logits[valid_mask]
            labels_valid = shift_labels[valid_mask]
            logits_f = logits_valid.float()
            log_denom = torch.logsumexp(logits_f, dim=-1)
            log_probs = logits_f - log_denom.unsqueeze(-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(-1)
            p_max = torch.exp(logits_f.max(dim=-1).values - log_denom)
            pred_ids = logits_valid.argmax(dim=-1)
            correct = pred_ids == labels_valid
            temp_logits = getattr(outputs, "temp_logits", None)
            temp_valid = None
            if temp_logits is not None:
                temp_valid = temp_logits[:, :-1, :].squeeze(-1)[valid_mask]
                used_temp_head = True
            finite_mask = torch.isfinite(entropy) & torch.isfinite(p_max)
            if temp_valid is not None:
                finite_mask &= torch.isfinite(temp_valid)
            if finite_mask.any():
                entropy_values.extend(entropy[finite_mask].detach().cpu().tolist())
                p_max_values.extend(p_max[finite_mask].detach().cpu().tolist())
                incorrect_values.extend((~correct[finite_mask]).to(torch.int).detach().cpu().tolist())
                if temp_valid is not None:
                    t_pred_values.extend(temp_valid[finite_mask].detach().cpu().tolist())
            processed_tokens += int(finite_mask.sum().item())
        processed_examples += attention_mask_t.size(0)

    if not entropy_values:
        raise RuntimeError("No valid tokens were collected. Check masks or dataset selection.")

    entropy_arr = np.asarray(entropy_values, dtype=np.float64)
    p_max_arr = np.asarray(p_max_values, dtype=np.float64)
    incorrect_arr = np.asarray(incorrect_values, dtype=np.int64)
    t_pred_arr = np.asarray(t_pred_values, dtype=np.float64) if used_temp_head else None

    results: Dict[str, Any] = {
        "dataset": args.dataset_name,
        "split": split_name,
        "text_field": text_field,
        "examples_processed": int(processed_examples),
        "tokens_processed": int(processed_tokens),
        "assistant_only": bool(args.assistant_only),
        "completion_only": bool(args.completion_only),
        "assistant_mask_used_examples": int(assistant_mask_used_examples),
        "assistant_mask_missing_examples": int(assistant_mask_missing_examples),
        "used_temp_head": bool(used_temp_head),
    }

    auroc_entropy = _roc_auc_score(incorrect_arr, entropy_arr)
    auroc_neg_pmax = _roc_auc_score(incorrect_arr, -p_max_arr)
    results["auroc_entropy"] = auroc_entropy
    results["auroc_neg_pmax"] = auroc_neg_pmax

    if used_temp_head and t_pred_arr is not None:
        auroc_tpred = _roc_auc_score(incorrect_arr, t_pred_arr)
        results["auroc_t_pred"] = auroc_tpred
    else:
        results["auroc_t_pred"] = None

    if args.confidence_quantile is not None:
        threshold = float(np.quantile(p_max_arr, args.confidence_quantile))
        threshold_mode = f"quantile={args.confidence_quantile}"
    else:
        threshold = float(args.confidence_threshold)
        threshold_mode = f"threshold={args.confidence_threshold}"

    confident_mask = p_max_arr > threshold
    confident_count = int(confident_mask.sum())
    confident_incorrect = int((incorrect_arr[confident_mask] == 1).sum())
    results["confidence_threshold"] = threshold
    results["confidence_threshold_mode"] = threshold_mode
    results["confident_token_count"] = confident_count
    results["confident_incorrect_count"] = confident_incorrect
    results["confident_incorrect_rate"] = (
        float(confident_incorrect) / confident_count if confident_count > 0 else None
    )

    auroc_entropy_confident = None
    auroc_neg_pmax_confident = None
    auroc_tpred_confident = None
    if confident_count > 0:
        auroc_entropy_confident = _roc_auc_score(
            incorrect_arr[confident_mask],
            entropy_arr[confident_mask],
        )
        auroc_neg_pmax_confident = _roc_auc_score(
            incorrect_arr[confident_mask],
            -p_max_arr[confident_mask],
        )
        if used_temp_head and t_pred_arr is not None:
            auroc_tpred_confident = _roc_auc_score(
                incorrect_arr[confident_mask],
                t_pred_arr[confident_mask],
            )
    results["auroc_entropy_confident"] = auroc_entropy_confident
    results["auroc_neg_pmax_confident"] = auroc_neg_pmax_confident
    results["auroc_t_pred_confident"] = auroc_tpred_confident

    print(f"AUROC T_pred: {results['auroc_t_pred']}")
    print(f"AUROC H:      {results['auroc_entropy']}")
    print(f"AUROC -Pmax:  {results['auroc_neg_pmax']}")
    if args.assistant_only and assistant_mask_missing_examples > 0:
        print(
            f"[!] assistant_only requested, but assistant masks were missing for "
            f"{assistant_mask_missing_examples} example(s). Falling back to unmasked tokens."
        )
    print(
        f"Confident tokens ({threshold_mode}): {confident_count} "
        f"incorrect={confident_incorrect} rate="
        f"{results['confident_incorrect_rate']}"
    )
    if confident_count > 0:
        print(f"AUROC H | confident:      {auroc_entropy_confident}")
        print(f"AUROC -Pmax | confident:  {auroc_neg_pmax_confident}")
        if used_temp_head:
            print(f"AUROC T_pred | confident: {auroc_tpred_confident}")
        if confident_count < args.min_confident_tokens:
            print(
                f"[!] confident token count ({confident_count}) is below --min_confident_tokens "
                f"({args.min_confident_tokens}); confident-only AUROCs may be noisy."
            )

    summary_path = os.path.join(args.output_dir, "token_signal_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    txt_path = os.path.join(args.output_dir, "token_signal_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
        f.write("\n")

    if used_temp_head and t_pred_arr is not None:
        total_points = len(entropy_arr)
        if args.max_scatter_points and total_points > args.max_scatter_points:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(total_points, size=args.max_scatter_points, replace=False)
            x_vals = entropy_arr[indices]
            y_vals = t_pred_arr[indices]
            colors = np.where(incorrect_arr[indices] == 1, "#D64545", "#2E86AB")
        else:
            x_vals = entropy_arr
            y_vals = t_pred_arr
            colors = np.where(incorrect_arr == 1, "#D64545", "#2E86AB")

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.scatter(x_vals, y_vals, c=colors, s=6, alpha=0.35, edgecolors="none")
        ax.set_xlabel("Entropy (nats)")
        ax.set_ylabel("T_pred")
        ax.set_title("Token Scatter: entropy vs T_pred")
        ax.grid(alpha=0.2)
        scatter_path = os.path.join(args.output_dir, "token_signal_scatter.png")
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=160)


if __name__ == "__main__":
    main()
