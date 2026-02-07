from typing import Any, Optional, Union
from dataclasses import dataclass

import torch
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy
import os
import json
import time
from safetensors.torch import save_file


import argparse

from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from model import AutoDecoModelForCausalLM
from trainer.trl_autodeco import AutoDecoLLMTrainer

# import sys
# PATH_TO_TRL="trl/"
# sys.path.append(PATH_TO_TRL)
import numpy as np

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
# from sft_trainer import SFTTrainer

# Custom ScriptArguments to add training head parameters
@dataclass
class AutoDecoLLMScriptArguments(ScriptArguments):
    """Script arguments for AutoDecoLLM training."""
    train_temp: bool = False
    train_top_p: bool = False
    temp_objective: str = "legacy_ce"
    min_p_ratio: float = 0.1
    temp_hinge_weight: float = 1.0
    temp_reg_weight: float = 0.0
    easy_token_drop_prob: float = 0.6
    goldilocks_filter: bool = False
    goldilocks_easy_frac: float = 0.1
    goldilocks_topk_frac: float = 0.9
    goldilocks_topk: int = 10

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"attention_mask"`: Tensor of attention mask, padded to the maximum length of the batch.
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly. The attention mask will be set to 1 for all tokens.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'position_ids': tensor([[0, 1, 2],
                             [0, 1, 0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]

        # Check if we have meaningful seq_lengths from packing (restarting sequences)
        has_packed_position_ids = self.return_position_ids and "seq_lengths" in examples[0] and self.padding_free

        # For packing with position_ids, we should NOT create attention_mask as it causes
        # flash attention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask
        if not has_packed_position_ids:
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "seq_lengths" in examples[0]:
                position_ids = self._convert_seq_lengths_to_position_ids(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if "temp_labels" in examples[0]:
            temp_labels = [torch.tensor(example["temp_labels"]) for example in examples]
            if "top_p_labels" in examples[0]:
                top_p = [torch.tensor(example["top_p_labels"]) for example in examples]
            if "top_k_labels" in examples[0]:
                top_k = [torch.tensor(example["top_k_labels"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # Pad
        output = {}
        if self.padding_free:
            output["input_ids"] = torch.cat(input_ids, dim=0).unsqueeze(0)
            if not has_packed_position_ids:
                output["attention_mask"] = torch.cat(attention_mask, dim=0).unsqueeze(0)
            if self.return_position_ids:
                output["position_ids"] = torch.cat(position_ids, dim=0).unsqueeze(0)
            output["labels"] = torch.cat(labels, dim=0).unsqueeze(0)
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = torch.cat(completion_mask, dim=0).unsqueeze(0)
                output["labels"][completion_mask == 0] = -100
            if "assistant_masks" in examples[0]:
                assistant_masks = torch.cat(assistant_masks, dim=0).unsqueeze(0)
                output["labels"][assistant_masks == 0] = -100
        else:
            output["input_ids"] = pad(
                input_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if self.return_position_ids:
                output["position_ids"] = pad(
                    position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
            output["labels"] = pad(
                labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            if "temp_labels" in examples[0]:
                temp_labels = pad(
                    temp_labels, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["temp_labels"] = temp_labels
            if "top_p_labels" in examples[0]:
                top_p = pad(
                    top_p, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["top_p_labels"] = top_p
            if "top_k_labels" in examples[0]:
                top_k = pad(
                    top_k, padding_value=-1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["top_k_labels"] = top_k
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = pad(
                    completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

            if "assistant_masks" in examples[0]:
                assistant_masks = pad(
                    assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
                )
                output["labels"][assistant_masks == 0] = -100
        return output


def _get_global_rank() -> int:
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank)
        except ValueError:
            pass
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError:
            pass
    return 0


def _prompt_choice(prompt: str, options: list[str], default: str) -> str:
    print(prompt)
    for idx, option in enumerate(options, 1):
        marker = " (default)" if option == default else ""
        print(f"  [{idx}] {option}{marker}")

    while True:
        try:
            user_input = input("Select by number or name (Enter for default): ").strip()
        except EOFError:
            print(f"[!] No interactive stdin detected. Using default: {default}")
            return default

        if user_input == "":
            return default
        if user_input.isdigit():
            index = int(user_input) - 1
            if 0 <= index < len(options):
                return options[index]
        if user_input in options:
            return user_input
        print("[!] Invalid selection. Please try again.")


def _resolve_local_dataset_file(dataset_name: str) -> Optional[str]:
    candidates = [dataset_name]
    if not os.path.isabs(dataset_name):
        candidates.append(os.path.join("data", dataset_name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _is_conversation_turn(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    role = item.get("role", None)
    content = item.get("content", None)
    if not isinstance(role, str):
        return False
    if not isinstance(content, str):
        return False
    return True


def _sanity_check_conversation_split(dataset_split, column_name: str, sample_size: int = 32) -> None:
    max_samples = min(len(dataset_split), sample_size)
    checked = 0
    valid = 0
    first_bad = None

    for idx in range(max_samples):
        row = dataset_split[idx]
        value = row.get(column_name, None)
        if value is None:
            continue
        checked += 1
        is_valid = isinstance(value, list) and len(value) > 0 and all(_is_conversation_turn(x) for x in value)
        if is_valid:
            valid += 1
        elif first_bad is None:
            first_bad = {
                "index": idx,
                "type": type(value).__name__,
                "preview": str(value)[:200],
            }

    if checked == 0:
        raise ValueError(
            f"Column '{column_name}' has no non-null rows in the sampled subset; cannot validate conversation format."
        )

    ratio = valid / checked
    print(
        f"[!] Conversation sanity check for '{column_name}': "
        f"{valid}/{checked} sampled rows match list[dict(role, content)]."
    )
    if ratio < 0.5:
        detail = f" First bad sample: {first_bad}" if first_bad is not None else ""
        raise ValueError(
            f"Column '{column_name}' does not look like conversation data (need list of role/content dicts).{detail}"
        )


def _chat_template_has_generation_blocks(chat_template: Any) -> bool:
    if isinstance(chat_template, dict):
        template_text = "\n".join(v for v in chat_template.values() if isinstance(v, str))
    elif isinstance(chat_template, str):
        template_text = chat_template
    else:
        return False
    return (
        "{% generation %}" in template_text
        or "{%- generation %}" in template_text
    ) and (
        "{% endgeneration %}" in template_text
        or "{%- endgeneration %}" in template_text
    )


def _probe_assistant_mask_support(
    tokenizer,
    dataset_split,
    text_field: str,
    sample_size: int = 16,
) -> tuple[bool, str]:
    if text_field not in {"messages", "conversations"}:
        return False, f"text field '{text_field}' is not conversational."

    max_samples = min(len(dataset_split), sample_size)
    checked = 0
    with_mask = 0
    with_positive_mask = 0
    first_error = None

    for idx in range(max_samples):
        row = dataset_split[idx]
        convo = row.get(text_field, None)
        if not (isinstance(convo, list) and len(convo) > 0):
            continue
        if not all(_is_conversation_turn(x) for x in convo):
            continue
        checked += 1
        try:
            processed = tokenizer.apply_chat_template(
                convo,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
            )
        except Exception as e:
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
            continue

        assistant_masks = processed.get("assistant_masks")
        if assistant_masks is None:
            continue
        with_mask += 1
        if any(int(x) == 1 for x in assistant_masks):
            with_positive_mask += 1

    if checked == 0:
        return False, "no conversational rows were probe-able."
    if with_positive_mask > 0:
        return True, (
            f"assistant masks available on {with_positive_mask}/{checked} probed rows "
            f"(mask key present on {with_mask}/{checked})."
        )

    detail = f" first error: {first_error}" if first_error is not None else ""
    return False, (
        f"assistant masks unavailable on {checked}/{checked} probed rows "
        f"(mask key present on {with_mask}/{checked}, positive masks on {with_positive_mask}/{checked}).{detail}"
    )


def _normalize_chat_split_for_trl(dataset_split, selected_text_field: str):
    """
    Normalize conversational datasets to avoid TRL conversational-detection ambiguity.

    Some datasets expose both `prompt` and `messages`. Older TRL checks can pick `prompt`
    first and incorrectly conclude the dataset is non-conversational. We canonicalize the
    selected conversational column to `messages` and drop conflicting prompt-completion keys.
    """
    if selected_text_field not in {"messages", "conversations"}:
        return dataset_split, selected_text_field

    normalized_field = selected_text_field
    if selected_text_field == "conversations":
        if "messages" in dataset_split.column_names:
            # Prefer existing canonical column if already present.
            normalized_field = "messages"
        else:
            dataset_split = dataset_split.rename_column("conversations", "messages")
            normalized_field = "messages"

    conflicting_cols = [
        col for col in ("prompt", "completion", "chosen", "rejected")
        if col in dataset_split.column_names
    ]
    if conflicting_cols:
        print(
            "[!] Dropping conflicting conversational keys for TRL compatibility: "
            + ", ".join(conflicting_cols)
        )
        dataset_split = dataset_split.remove_columns(conflicting_cols)

    return dataset_split, normalized_field


def _load_and_prepare_dataset(script_args, training_args):
    load_start = time.time()
    dataset_name = script_args.dataset_name
    dataset_config = getattr(script_args, "dataset_config", None)
    if dataset_config is None:
        dataset_config = getattr(script_args, "dataset_config_name", None)
    local_data_file = _resolve_local_dataset_file(dataset_name)

    if local_data_file is not None:
        print(f"[!] Loading local JSON dataset: {local_data_file}")
        dataset = load_dataset("json", data_files=local_data_file)
    else:
        print(f"[!] Loading Hugging Face dataset: {dataset_name}")
        try:
            if dataset_config is not None:
                print(f"[!] Using HF dataset config: {dataset_config}")
                dataset = load_dataset(dataset_name, dataset_config)
            else:
                dataset = load_dataset(dataset_name)
        except Exception as e:
            raise ValueError(
                f"Failed to load HF dataset '{dataset_name}'. "
                "If it requires a config, pass --dataset_config or --dataset_config_name."
            ) from e

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    split_names = list(dataset.keys())
    if len(split_names) == 0:
        raise ValueError("Loaded dataset has no splits.")

    default_split = script_args.dataset_train_split if script_args.dataset_train_split in split_names else split_names[0]
    selection_cache = os.path.join(training_args.output_dir, ".dataset_selection.json")
    rank = _get_global_rank()
    is_main = rank == 0

    os.makedirs(training_args.output_dir, exist_ok=True)
    if is_main and os.path.exists(selection_cache):
        os.remove(selection_cache)

    if is_main:
        if len(split_names) == 1:
            selected_split = split_names[0]
            print(f"[!] Only one split found. Using split: {selected_split}")
        else:
            selected_split = _prompt_choice("Choose dataset split for training:", split_names, default_split)

        split_dataset = dataset[selected_split]
        column_names = list(split_dataset.column_names)
        if len(column_names) == 0:
            raise ValueError(f"Split '{selected_split}' has no columns.")

        if "messages" in column_names:
            selected_text_field = "messages"
            print("[!] Found 'messages' column. Using it for conversational training.")
        elif "conversations" in column_names:
            selected_text_field = "conversations"
            print("[!] Found 'conversations' column. Using it for conversational training.")
        else:
            default_text_field = (
                script_args.dataset_text_field
                if script_args.dataset_text_field in column_names
                else column_names[0]
            )
            selected_text_field = _prompt_choice(
                f"Choose text column from split '{selected_split}':",
                column_names,
                default_text_field,
            )

        if selected_text_field in {"messages", "conversations"}:
            _sanity_check_conversation_split(split_dataset, selected_text_field)

        selection = {
            "split": selected_split,
            "text_field": selected_text_field,
        }
        with open(selection_cache, "w", encoding="utf-8") as f:
            json.dump(selection, f)
    else:
        timeout_s = 300
        start = time.time()
        while True:
            if os.path.exists(selection_cache):
                mtime = os.path.getmtime(selection_cache)
                if mtime >= load_start - 1.0:
                    break
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for dataset selection file '{selection_cache}'."
                )
            time.sleep(0.5)
        with open(selection_cache, "r", encoding="utf-8") as f:
            selection = json.load(f)

    selected_split = selection["split"]
    selected_text_field = selection["text_field"]

    split_dataset = dataset[selected_split]
    split_dataset, selected_text_field = _normalize_chat_split_for_trl(
        split_dataset, selected_text_field
    )
    dataset[selected_split] = split_dataset

    script_args.dataset_train_split = selected_split
    script_args.dataset_text_field = selected_text_field
    print(
        f"[!] Final dataset selection: split='{script_args.dataset_train_split}', "
        f"text_field='{script_args.dataset_text_field}'"
    )
    return dataset


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )

    # If we only train heads, avoid saving full model checkpoints during training
    if script_args.train_temp or script_args.train_top_p:
        try:
            training_args.save_strategy = "no"
        except Exception:
            pass

    valid_temp_objectives = {"legacy_ce", "analytic_min_p_hinge"}
    if script_args.temp_objective not in valid_temp_objectives:
        raise ValueError(
            f"temp_objective must be one of {sorted(valid_temp_objectives)}, got {script_args.temp_objective}"
        )
    if not (0.0 < script_args.min_p_ratio < 1.0):
        raise ValueError(f"min_p_ratio must be in (0, 1), got {script_args.min_p_ratio}")
    if not (0.0 <= script_args.easy_token_drop_prob <= 1.0):
        raise ValueError(
            f"easy_token_drop_prob must be in [0, 1], got {script_args.easy_token_drop_prob}"
        )
    if not (0.0 <= script_args.goldilocks_easy_frac <= 1.0):
        raise ValueError(
            f"goldilocks_easy_frac must be in [0, 1], got {script_args.goldilocks_easy_frac}"
        )
    if not (0.0 <= script_args.goldilocks_topk_frac <= 1.0):
        raise ValueError(
            f"goldilocks_topk_frac must be in [0, 1], got {script_args.goldilocks_topk_frac}"
        )
    if (script_args.goldilocks_easy_frac + script_args.goldilocks_topk_frac) <= 0.0:
        raise ValueError(
            "goldilocks_easy_frac + goldilocks_topk_frac must be > 0."
        )
    if script_args.goldilocks_topk < 1:
        raise ValueError(f"goldilocks_topk must be >= 1, got {script_args.goldilocks_topk}")

    model = AutoDecoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # Sync model-internal training flags with CLI selection.
    model.train_temp = script_args.train_temp
    model.train_top_p = script_args.train_top_p
    if hasattr(model, "config"):
        model.config.enable_temperature_head = script_args.train_temp
        model.config.enable_top_p_head = script_args.train_top_p

    # Configure which heads to train based on script arguments
    if script_args.train_temp or script_args.train_top_p:
        print(f"[!] Training configuration: train_temp={script_args.train_temp}, train_top_p={script_args.train_top_p}")
        if script_args.train_temp:
            print(
                f"[!] Temperature objective: {script_args.temp_objective} "
                f"(min_p_ratio={script_args.min_p_ratio}, "
                f"temp_hinge_weight={script_args.temp_hinge_weight}, "
                f"temp_reg_weight={script_args.temp_reg_weight}, "
                f"easy_token_drop_prob={script_args.easy_token_drop_prob}, "
                f"goldilocks_filter={script_args.goldilocks_filter}, "
                f"goldilocks_easy_frac={script_args.goldilocks_easy_frac}, "
                f"goldilocks_topk_frac={script_args.goldilocks_topk_frac}, "
                f"goldilocks_topk={script_args.goldilocks_topk})"
            )
    else:
        print(f"[!] Training the LLM model itself. No AutoDeco training.")
    for name, param in model.named_parameters():
        if 'temp_head' in name:
            param.requires_grad = script_args.train_temp
            status = "Training" if script_args.train_temp else "Freezing"
            print(f"[!] {status} parameter: {name}")
            continue
        elif 'top_p_head' in name:
            param.requires_grad = script_args.train_top_p
            status = "Training" if script_args.train_top_p else "Freezing"
            print(f"[!] {status} parameter: {name}")
            continue
        else:
            if not script_args.train_temp and not script_args.train_top_p:
                param.requires_grad = True
                print(f"[!] Training parameter: {name}")
            else:
                param.requires_grad = False
                print(f"[!] Freezing parameter: {name}")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True, 
    )
    # Set default chat template if needed
    if tokenizer.chat_template is None:
        print("[!] Tokenizer has no chat_template. Applying fallback ChatML template.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # Ensure PAD token exists (some decoder-only models like LLaMA don't define one)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id

    dataset = _load_and_prepare_dataset(script_args, training_args)

    if training_args.assistant_only_loss:
        has_generation_blocks = _chat_template_has_generation_blocks(tokenizer.chat_template)
        if not has_generation_blocks:
            print(
                "[!] assistant_only_loss=True but tokenizer chat template does not visibly include "
                "{% generation %}/{% endgeneration %} blocks."
            )

        train_split = dataset[script_args.dataset_train_split]
        assistant_mask_supported, assistant_mask_reason = _probe_assistant_mask_support(
            tokenizer=tokenizer,
            dataset_split=train_split,
            text_field=script_args.dataset_text_field,
        )
        if assistant_mask_supported:
            print(f"[!] Assistant-mask preflight passed: {assistant_mask_reason}")
        else:
            print(f"[!] Assistant-mask preflight failed: {assistant_mask_reason}")
            print(
                "[!] Disabling assistant_only_loss for this run to avoid tokenization failure. "
                "Training will proceed without assistant masking."
            )
            training_args.assistant_only_loss = False

    # FOR DEBUGGING
    # dataset['train'] = dataset['train'].select(range(50))

    if script_args.train_temp or script_args.train_top_p:
        print(f"[!] AutoDecoLLM Training")
        trainer = AutoDecoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=training_args.completion_only_loss,
            padding_free=training_args.padding_free,
            pad_to_multiple_of=training_args.pad_to_multiple_of,
        ),
        peft_config=get_peft_config(model_args),
        temp_objective=script_args.temp_objective,
        min_p_ratio=script_args.min_p_ratio,
        temp_hinge_weight=script_args.temp_hinge_weight,
        temp_reg_weight=script_args.temp_reg_weight,
        easy_token_drop_prob=script_args.easy_token_drop_prob,
        goldilocks_filter=script_args.goldilocks_filter,
        goldilocks_easy_frac=script_args.goldilocks_easy_frac,
        goldilocks_topk_frac=script_args.goldilocks_topk_frac,
        goldilocks_topk=script_args.goldilocks_topk,
        )
    else:
        print(f"[!] Normal SFT Training")
        eval_dataset = None
        if training_args.eval_strategy != "no":
            if script_args.dataset_test_split in dataset:
                eval_dataset = dataset[script_args.dataset_test_split]
            else:
                print(
                    f"[!] Requested eval split '{script_args.dataset_test_split}' not found. "
                    f"Available splits: {list(dataset.keys())}. Skipping eval dataset."
                )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=eval_dataset,
            peft_config=get_peft_config(model_args),
        )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (AutoDecoLLMScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
