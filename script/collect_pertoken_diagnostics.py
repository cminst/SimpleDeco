"""
Collect per-token diagnostics for AutoDeco heads using teacher forcing.

Stores a HuggingFace Dataset (save_to_disk) containing:
  - per-token AutoDeco outputs (T_hat, p_hat)
  - logit-shape features at T=1 (entropy, p_max, gap12, top-k mass, expH)
  - optional top-K sketch (ids + logits)
  - optional nucleus size under (T_hat, p_hat)
  - token/context indicators (newline, punct, whitespace, boundary, code block)
  - per-sequence metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.templlm_auto import AutoDecoModelForCausalLM

_PUNCT = set(string.punctuation)
_BOUNDARY_CHARS = {".", "?", "!", "\n"}


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
) -> Tuple[List[int], List[int] | None, bool, int, int]:
    prompt_value = row.get("prompt")
    completion_value = row.get("completion")

    output_ids: List[int]
    label_mask: List[int] | None = None
    assistant_mask_used = False
    prompt_len = 0
    completion_len = 0

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
        prompt_len = len(prompt_ids)
        completion_len = max(len(output_ids) - len(prompt_ids), 0)
        completion_mask = [0] * len(prompt_ids) + [1] * completion_len
        if assistant_only:
            assistant_masks = prompt_completion.get("assistant_masks")
            if assistant_masks is not None and len(assistant_masks) == len(output_ids):
                label_mask = [int(x) for x in assistant_masks]
                assistant_mask_used = True
        if label_mask is None and completion_only:
            label_mask = completion_mask
        return output_ids, label_mask, assistant_mask_used, prompt_len, completion_len

    prompt_text = prompt_value if isinstance(prompt_value, str) else str(prompt_value)
    completion_text = completion_value if isinstance(completion_value, str) else str(completion_value)
    prompt_ids = tokenizer(text=prompt_text)["input_ids"]
    output_ids = tokenizer(text=prompt_text + completion_text)["input_ids"]
    if tokenizer.bos_token_id is not None:
        if prompt_ids and prompt_ids[0] == tokenizer.bos_token_id:
            prompt_ids = prompt_ids[1:]
        if output_ids and output_ids[0] == tokenizer.bos_token_id:
            output_ids = output_ids[1:]
    prompt_len = len(prompt_ids)
    completion_len = max(len(output_ids) - len(prompt_ids), 0)
    completion_mask = [0] * len(prompt_ids) + [1] * completion_len
    if completion_only:
        label_mask = completion_mask
    return output_ids, label_mask, assistant_mask_used, prompt_len, completion_len


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
) -> Tuple[List[int], List[int] | None, bool, int, int]:
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
                label_mask = [int(x) for x in assistant_masks]
                prompt_len = len(label_mask) - sum(label_mask)
                completion_len = sum(label_mask)
                return input_ids, label_mask, True, prompt_len, completion_len
        return input_ids, None, False, len(input_ids), 0

    value = row.get(text_field)
    if value is None:
        raise KeyError(f"Row does not contain field '{text_field}'.")
    if not isinstance(value, str):
        value = str(value)
    if user_suffix:
        value = f"{value}{user_suffix}"
    input_ids = tokenizer(text=value)["input_ids"]
    return input_ids, None, False, len(input_ids), 0


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


def _parse_int_list(value: str | None, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return list(default)
    return [int(p) for p in parts]


def _stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _compute_token_indicators(input_ids: List[int], tokenizer: AutoTokenizer) -> Dict[str, List[bool]]:
    indicators = {
        "is_newline": [],
        "is_punct": [],
        "is_whitespace": [],
        "is_boundary": [],
        "in_code_block": [],
    }
    decoded_text = ""
    prev_tail = ""
    in_code_block = False
    for token_id in input_ids:
        token_text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        prev_char = decoded_text[-1] if decoded_text else ""
        stripped = token_text.strip()
        indicators["is_newline"].append("\n" in token_text)
        indicators["is_whitespace"].append(bool(token_text) and len(stripped) == 0)
        indicators["is_punct"].append(bool(stripped) and all(ch in _PUNCT for ch in stripped))
        indicators["is_boundary"].append(prev_char in _BOUNDARY_CHARS)
        indicators["in_code_block"].append(in_code_block)

        chunk = prev_tail + token_text
        if chunk.count("```") % 2 == 1:
            in_code_block = not in_code_block
        decoded_text += token_text
        prev_tail = (prev_tail + token_text)[-2:]
    return indicators


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-token AutoDeco diagnostics and save a HuggingFace Dataset.",
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
        help="Random seed used for shuffling.",
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
        "--topk_mass",
        default="10,50,200",
        help="Comma-separated list of top-k mass cutoffs to compute.",
    )
    parser.add_argument(
        "--topk_sketch",
        type=int,
        default=128,
        help="Store top-K token ids/logits per token; set 0 to disable.",
    )
    parser.add_argument(
        "--compute_nucleus_size",
        action="store_true",
        help="Compute nucleus size under (T_hat, p_hat) using top-k approximation.",
    )
    parser.add_argument(
        "--nucleus_max_k",
        type=int,
        default=1024,
        help="Max k for nucleus size approximation (top-k).",
    )
    parser.add_argument(
        "--base_temperature",
        type=float,
        default=None,
        help="Optional baseline temperature recorded in metadata.",
    )
    parser.add_argument(
        "--base_top_p",
        type=float,
        default=None,
        help="Optional baseline top-p recorded in metadata.",
    )
    parser.add_argument(
        "--base_top_k",
        type=int,
        default=None,
        help="Optional baseline top-k recorded in metadata.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the saved dataset.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Token rows per in-memory chunk before flushing to Dataset.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Progress logging frequency in processed examples; use <=0 to disable.",
    )
    args = parser.parse_args()

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

    collect_temp = hasattr(model, "temp_head")
    collect_top_p = bool(getattr(model, "train_top_p", False))
    heads = []
    if collect_temp:
        heads.append("temp_head")
    if collect_top_p:
        heads.append("top_p_head")
    if not heads:
        heads = ["none"]
    print(f"[collect_pertoken] Collecting heads: {', '.join(heads)}")

    vocab_size = getattr(getattr(model, "llm", None), "config", None)
    vocab_size = getattr(vocab_size, "vocab_size", None) or getattr(model.config, "vocab_size", None)

    mass_k_list = _parse_int_list(args.topk_mass, [10, 50, 200])
    mass_k_list = sorted({k for k in mass_k_list if k > 0})

    topk_sketch = max(int(args.topk_sketch), 0)
    topk_mass_k = max(mass_k_list + ([topk_sketch] if topk_sketch > 0 else [])) if mass_k_list or topk_sketch > 0 else 0

    method_payload = {
        "model_name_or_path": args.model_name_or_path,
        "base_temperature": args.base_temperature,
        "base_top_p": args.base_top_p,
        "base_top_k": args.base_top_k,
        "enable_temperature_head": bool(getattr(model.config, "enable_temperature_head", True)),
        "enable_top_p_head": bool(getattr(model.config, "enable_top_p_head", False)),
    }
    method_id = _hash_payload(method_payload)[:12]
    config_hash = _hash_payload(getattr(model.config, "to_dict", lambda: {})())

    total_examples = len(dataset_split)
    indices = list(range(total_examples))
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(indices)
    if args.max_examples is not None:
        indices = indices[: args.max_examples]

    token_rows: List[Dict[str, Any]] = []
    token_datasets: List[Dataset] = []
    seq_rows: List[Dict[str, Any]] = []
    seq_token_counts: Dict[int, int] = {}

    assistant_mask_used_examples = 0
    assistant_mask_missing_examples = 0

    processed_examples = 0
    processed_tokens = 0
    seq_id_counter = 0

    batch: List[Dict[str, Any]] = []

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
            raise RuntimeError("Model did not return logits for per-token diagnostics.")

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return

        logits_valid = shift_logits[valid_mask]
        labels_valid = shift_labels[valid_mask]

        logits_f = logits_valid.float()
        log_denom = torch.logsumexp(logits_f, dim=-1)
        log_probs = logits_f - log_denom.unsqueeze(-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(-1)
        max_logit = logits_f.max(dim=-1).values
        p_max = torch.exp(max_logit - log_denom)
        expH = torch.exp(entropy)

        vocab = logits_f.size(-1)
        log_vocab = math.log(vocab) if vocab > 0 else 1.0
        h_norm = entropy / log_vocab

        top2_logits, _ = torch.topk(logits_f, k=min(2, vocab), dim=-1)
        if top2_logits.size(-1) == 2:
            gap12 = top2_logits[:, 0] - top2_logits[:, 1]
        else:
            gap12 = torch.zeros_like(entropy)

        topk_ids = None
        topk_logits = None
        mass_values: Dict[int, torch.Tensor] = {}
        if topk_mass_k > 0:
            k = min(topk_mass_k, vocab)
            topk_logits, topk_ids = torch.topk(logits_f, k=k, dim=-1)
            topk_probs = torch.exp(topk_logits - log_denom.unsqueeze(-1))
            for mass_k in mass_k_list:
                kk = min(mass_k, k)
                mass_values[mass_k] = topk_probs[:, :kk].sum(-1)

        temp_logits = getattr(outputs, "temp_logits", None)
        temp_valid = None
        if collect_temp and temp_logits is not None:
            temp_valid = temp_logits[:, :-1, :].squeeze(-1)[valid_mask]

        top_p_valid = None
        if collect_top_p:
            top_p_logits = getattr(outputs, "top_p_logits", None)
            if top_p_logits is not None:
                top_p_valid = top_p_logits[:, :-1, :].squeeze(-1)[valid_mask]

        positions = torch.arange(1, shift_labels.size(1) + 1, device=shift_labels.device)
        positions_flat = positions.unsqueeze(0).expand(shift_labels.size(0), -1)[valid_mask]

        finite_mask = torch.isfinite(entropy) & torch.isfinite(p_max) & torch.isfinite(gap12)
        if temp_valid is not None:
            finite_mask &= torch.isfinite(temp_valid)
        if top_p_valid is not None:
            finite_mask &= torch.isfinite(top_p_valid)

        mask_list = finite_mask.detach().cpu().tolist()

        entropy_list = entropy[finite_mask].detach().cpu().tolist()
        h_norm_list = h_norm[finite_mask].detach().cpu().tolist()
        p_max_list = p_max[finite_mask].detach().cpu().tolist()
        gap12_list = gap12[finite_mask].detach().cpu().tolist()
        expH_list = expH[finite_mask].detach().cpu().tolist()
        token_id_list = labels_valid[finite_mask].detach().cpu().tolist()
        pos_list = positions_flat[finite_mask].detach().cpu().tolist()

        temp_list: List[float | None]
        if temp_valid is not None:
            temp_list = temp_valid[finite_mask].detach().cpu().tolist()
        else:
            temp_list = [None] * len(entropy_list)

        p_hat_list: List[float | None]
        if top_p_valid is not None:
            p_hat_list = top_p_valid[finite_mask].detach().cpu().tolist()
        else:
            p_hat_list = [None] * len(entropy_list)

        mass_lists: Dict[int, List[float]] = {}
        for mass_k, tensor_vals in mass_values.items():
            mass_lists[mass_k] = tensor_vals[finite_mask].detach().cpu().tolist()
        for mass_k in mass_k_list:
            mass_lists.setdefault(mass_k, [None] * len(entropy_list))

        topk_ids_list: List[List[int]] = []
        topk_logits_list: List[List[float]] = []
        if topk_sketch > 0 and topk_ids is not None and topk_logits is not None:
            k = min(topk_sketch, topk_ids.size(-1))
            sketch_ids = topk_ids[:, :k].to(torch.int32).detach().cpu().tolist()
            sketch_logits = topk_logits[:, :k].to(torch.float16).detach().cpu().tolist()
            for ids, logits_vals, keep in zip(sketch_ids, sketch_logits, mask_list):
                if keep:
                    topk_ids_list.append(ids)
                    topk_logits_list.append(logits_vals)
        else:
            topk_ids_list = [[] for _ in entropy_list]
            topk_logits_list = [[] for _ in entropy_list]

        nucleus_size_list: List[int | None] = [None] * len(entropy_list)
        if args.compute_nucleus_size and temp_valid is not None and top_p_valid is not None:
            temp_clamped = temp_valid.clamp_min(1e-4)
            scaled_logits = logits_f / temp_clamped.unsqueeze(-1)
            log_denom_scaled = torch.logsumexp(scaled_logits, dim=-1)
            k = min(max(int(args.nucleus_max_k), 1), vocab)
            topk_scaled_logits, _ = torch.topk(scaled_logits, k=k, dim=-1)
            topk_scaled_probs = torch.exp(topk_scaled_logits - log_denom_scaled.unsqueeze(-1))
            cum_probs = torch.cumsum(topk_scaled_probs, dim=-1)
            p_hat_clamped = top_p_valid.clamp(0.0, 1.0)
            nucleus_sizes = torch.searchsorted(cum_probs, p_hat_clamped.unsqueeze(-1), right=False).squeeze(-1)
            nucleus_sizes = nucleus_sizes + 1
            not_reached = p_hat_clamped > cum_probs[:, -1]
            nucleus_sizes = torch.where(not_reached, torch.full_like(nucleus_sizes, -1), nucleus_sizes)
            nucleus_sizes_list = nucleus_sizes[finite_mask].detach().cpu().tolist()
            nucleus_size_list = [int(x) for x in nucleus_sizes_list]

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
            seq_id = example["seq_id"]
            if count == 0:
                seq_token_counts[seq_id] = seq_token_counts.get(seq_id, 0)
                continue
            indicators = _compute_token_indicators(example["input_ids"], tokenizer)
            for idx in range(count):
                pos = int(pos_list[token_offset + idx])
                mass_fields = {f"mass{k}": mass_lists[k][token_offset + idx] for k in mass_k_list}
                token_rows.append(
                    {
                        "seq_id": int(seq_id),
                        "t": pos,
                        "token_id": int(token_id_list[token_offset + idx]),
                        "T_hat": temp_list[token_offset + idx],
                        "p_hat": p_hat_list[token_offset + idx],
                        "method_id": method_id,
                        "H": entropy_list[token_offset + idx],
                        "H_norm": h_norm_list[token_offset + idx],
                        "p_max": p_max_list[token_offset + idx],
                        "gap12": gap12_list[token_offset + idx],
                        **mass_fields,
                        "expH": expH_list[token_offset + idx],
                        "nucleus_size_hat": nucleus_size_list[token_offset + idx],
                        "topk_ids": topk_ids_list[token_offset + idx],
                        "topk_logits": topk_logits_list[token_offset + idx],
                        "is_newline": bool(indicators["is_newline"][pos]),
                        "is_punct": bool(indicators["is_punct"][pos]),
                        "is_whitespace": bool(indicators["is_whitespace"][pos]),
                        "is_boundary": bool(indicators["is_boundary"][pos]),
                        "in_code_block": bool(indicators["in_code_block"][pos]),
                    }
                )
            token_offset += count
            seq_token_counts[seq_id] = seq_token_counts.get(seq_id, 0) + count

        processed_tokens += len(entropy_list)

    token_features = Features(
        {
            "seq_id": Value("int64"),
            "t": Value("int32"),
            "token_id": Value("int32"),
            "T_hat": Value("float32"),
            "p_hat": Value("float32"),
            "method_id": Value("string"),
            "H": Value("float32"),
            "H_norm": Value("float32"),
            "p_max": Value("float32"),
            "gap12": Value("float32"),
            **{f"mass{k}": Value("float32") for k in mass_k_list},
            "expH": Value("float32"),
            "nucleus_size_hat": Value("int32"),
            "topk_ids": Sequence(Value("int32")),
            "topk_logits": Sequence(Value("float16")),
            "is_newline": Value("bool"),
            "is_punct": Value("bool"),
            "is_whitespace": Value("bool"),
            "is_boundary": Value("bool"),
            "in_code_block": Value("bool"),
        }
    )

    for idx in indices:
        row = dataset_split[int(idx)]
        input_ids, label_mask, assistant_mask_used, prompt_len, completion_len = _tokenize_row(
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

        prompt_ids = input_ids[:prompt_len] if prompt_len > 0 else input_ids
        prompt_hash = hashlib.sha256(json.dumps(prompt_ids).encode("utf-8")).hexdigest()
        seq_len = len(input_ids)

        seq_id = seq_id_counter
        seq_id_counter += 1
        seq_rows.append(
            {
                "seq_id": int(seq_id),
                "dataset_index": int(idx),
                "task": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": split_name,
                "model_id": args.model_name_or_path,
                "autodeco_ckpt": args.model_name_or_path,
                "method_id": method_id,
                "config_hash": config_hash,
                "seed": int(args.seed),
                "prompt_hash": prompt_hash,
                "prompt_len": int(prompt_len),
                "gen_len": int(completion_len),
                "seq_len": int(seq_len),
                "assistant_only": bool(args.assistant_only),
                "completion_only": bool(args.completion_only),
                "assistant_mask_used": bool(assistant_mask_used),
            }
        )
        batch.append({"input_ids": input_ids, "label_mask": label_mask, "seq_id": seq_id})
        if len(batch) < args.batch_size:
            continue

        batch_size = len(batch)
        process_batch(batch)
        batch = []
        processed_examples += batch_size

        if args.max_tokens is not None and processed_tokens >= args.max_tokens:
            break
        if args.log_every > 0 and processed_examples % args.log_every == 0:
            print(f"[{processed_examples}/{len(indices)}] tokens={processed_tokens}")
        if args.chunk_size > 0 and len(token_rows) >= args.chunk_size:
            flush_tokens()

    if batch and (args.max_tokens is None or processed_tokens < args.max_tokens):
        batch_size = len(batch)
        process_batch(batch)
        processed_examples += batch_size

    if token_rows:
        flush_tokens()

    if not token_datasets:
        raise RuntimeError("No valid tokens were collected. Check masks or dataset selection.")

    tokens_dataset = token_datasets[0] if len(token_datasets) == 1 else concatenate_datasets(token_datasets)

    for row in seq_rows:
        seq_id = row["seq_id"]
        row["token_count"] = int(seq_token_counts.get(seq_id, 0))

    sequence_features = Features(
        {
            "seq_id": Value("int64"),
            "dataset_index": Value("int64"),
            "task": Value("string"),
            "dataset_config": Value("string"),
            "split": Value("string"),
            "model_id": Value("string"),
            "autodeco_ckpt": Value("string"),
            "method_id": Value("string"),
            "config_hash": Value("string"),
            "seed": Value("int32"),
            "prompt_hash": Value("string"),
            "prompt_len": Value("int32"),
            "gen_len": Value("int32"),
            "seq_len": Value("int32"),
            "assistant_only": Value("bool"),
            "completion_only": Value("bool"),
            "assistant_mask_used": Value("bool"),
            "token_count": Value("int32"),
        }
    )
    sequences_dataset = Dataset.from_list(seq_rows, features=sequence_features)

    DatasetDict({"tokens": tokens_dataset, "sequences": sequences_dataset}).save_to_disk(args.output_dir)

    if args.assistant_only and assistant_mask_missing_examples > 0:
        print(
            f"[!] assistant_only requested, but assistant masks were missing for "
            f"{assistant_mask_missing_examples} example(s). Falling back to unmasked tokens."
        )

    metadata = {
        "dataset": args.dataset_name,
        "split": split_name,
        "text_field": text_field,
        "examples_processed": int(processed_examples),
        "tokens_processed": int(processed_tokens),
        "collect_temp_head": bool(collect_temp),
        "collect_top_p_head": bool(collect_top_p),
        "topk_mass": mass_k_list,
        "topk_sketch": int(topk_sketch),
        "compute_nucleus_size": bool(args.compute_nucleus_size),
        "nucleus_max_k": int(args.nucleus_max_k),
        "method_id": method_id,
        "config_hash": config_hash,
        "base_temperature": args.base_temperature,
        "base_top_p": args.base_top_p,
        "base_top_k": args.base_top_k,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
