from trl import SFTTrainer
from transformers.trainer import _is_peft_model
import torch.nn.functional as F
import torch
import inspect
import json
import math
import os
from typing import Any, Callable, Optional, TypeVar, Union
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from accelerate import PartialState
import warnings
from trl import SFTConfig
from trl.trainer.sft_trainer import remove_none_values
from trl.data_utils import (
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    truncate_dataset,
)

try:
    from rich import print as rich_print
    from rich.markup import escape as rich_escape
    _RICH_AVAILABLE = True
except Exception:
    rich_print = None
    rich_escape = None
    _RICH_AVAILABLE = False

class AutoDecoLLMTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        pad_token_id: int = None,
        temp_loss_weight: float = 1.0,
        temp_objective: str = "legacy_ce",
        min_p_ratio: float = 0.1,
        temp_hinge_weight: float = 1.0,
        temp_reg_weight: float = 0.0,
        goldilocks_temp_cap: float = 2.0,
        goldilocks_uniform: bool = False,
        goldilocks_uniform_bins: int = 20,
        temp_target_smooth_window: int = 0,
        easy_token_drop_prob: float = 0.6,
        top_p_loss_method: str = "soft",
        temp_diag_enabled: bool = False,
        temp_diag_steps: int = 100,
        temp_diag_examples: int = 3,
        temp_diag_topk: int = 5,
        temp_diag_dir: str = "temp_diagnostics",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.temp_loss_weight = temp_loss_weight
        self.temp_objective = temp_objective
        self.min_p_ratio = min_p_ratio
        self.temp_hinge_weight = temp_hinge_weight
        self.temp_reg_weight = temp_reg_weight
        self.goldilocks_temp_cap = goldilocks_temp_cap
        self.goldilocks_uniform = bool(goldilocks_uniform)
        self.goldilocks_uniform_bins = max(1, int(goldilocks_uniform_bins))
        self.temp_target_smooth_window = max(0, int(temp_target_smooth_window))
        self.easy_token_drop_prob = easy_token_drop_prob
        self.top_p_loss_method = top_p_loss_method
        self.temp_diag_enabled = temp_diag_enabled
        self.temp_diag_steps = max(1, int(temp_diag_steps))
        self.temp_diag_examples = max(1, int(temp_diag_examples))
        self.temp_diag_topk = max(1, int(temp_diag_topk))
        self.temp_diag_dir = temp_diag_dir
        self._last_temp_diag_step = None
        self._temp_diag_output_dir = os.path.join(self.args.output_dir, self.temp_diag_dir)
        if self.temp_diag_enabled and self.is_world_process_zero():
            os.makedirs(self._temp_diag_output_dir, exist_ok=True)

    def _is_main_process_safe(self) -> bool:
        args = getattr(self, "args", None)
        if args is not None:
            return args.process_index == 0
        try:
            return PartialState().is_main_process
        except Exception:
            return True

    def _console_print(self, message: str) -> None:
        if _RICH_AVAILABLE and rich_print is not None:
            rich_print(message)
            return
        print(message)

    def _render_row_with_bold_assistant_tokens(
        self,
        input_ids: list[int],
        assistant_masks: list[int],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
    ) -> str:
        pieces: list[str] = []
        for token_id, mask_value in zip(input_ids, assistant_masks):
            token_text = processing_class.decode([int(token_id)], skip_special_tokens=False)
            if _RICH_AVAILABLE and rich_escape is not None:
                token_text = rich_escape(token_text)
            if int(mask_value) == 1:
                if _RICH_AVAILABLE:
                    pieces.append(f"[bold]{token_text}[/bold]")
                else:
                    pieces.append(token_text)
            else:
                pieces.append(token_text)
        return "".join(pieces)

    def _filter_no_assistant_token_rows(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        if not args.assistant_only_loss:
            return dataset
        if "assistant_masks" not in dataset.column_names:
            raise ValueError(
                "assistant_only_loss=True but dataset has no 'assistant_masks' column after preprocessing."
            )

        def _has_assistant_tokens(example: dict[str, Any]) -> bool:
            mask_values = example.get("assistant_masks")
            if not isinstance(mask_values, list):
                return False
            return any(int(x) == 1 for x in mask_values)

        filter_kwargs: dict[str, Any] = {}
        if isinstance(dataset, Dataset):
            total_before = len(dataset)
            filter_kwargs["num_proc"] = args.dataset_num_proc
            filter_kwargs["desc"] = f"Filtering {dataset_name} rows without assistant tokens"
            dataset = dataset.filter(_has_assistant_tokens, **filter_kwargs)
            total_after = len(dataset)
            removed = total_before - total_after
            if self._is_main_process_safe():
                self._console_print(
                    f"[!] assistant_only_loss filter: removed {removed} / {total_before} "
                    f"examples without assistant tokens; kept {total_after}."
                )
                if total_after > 0:
                    sample = dataset[0]
                    rendered = self._render_row_with_bold_assistant_tokens(
                        input_ids=sample["input_ids"],
                        assistant_masks=sample["assistant_masks"],
                        processing_class=processing_class,
                    )
                    self._console_print("[!] Example row with assistant tokens in bold:")
                    self._console_print(rendered)
            if total_after == 0:
                raise ValueError(
                    "assistant_only_loss=True but every example has zero assistant tokens after preprocessing."
                )
            return dataset

        dataset = dataset.filter(_has_assistant_tokens)
        if self._is_main_process_safe():
            self._console_print(
                "[!] assistant_only_loss filter applied to IterableDataset; exact removed/total counts are unavailable."
            )
        return dataset

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            if not args.dataset_num_proc:
                args.dataset_num_proc = os.cpu_count()
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    def _looks_like_chat_messages(value):
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

                    prompt_value = example.get("prompt")
                    completion_value = example.get("completion")
                    has_prompt_completion = prompt_value is not None and completion_value is not None

                    if has_prompt_completion:  # prompt-completion case
                        output = {}
                        if _looks_like_chat_messages(prompt_value) and _looks_like_chat_messages(completion_value):
                            prompt_ids = processing_class.apply_chat_template(
                                prompt_value,
                                tools=example.get("tools"),
                                tokenize=True,
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_processed = processing_class.apply_chat_template(
                                prompt_value + completion_value,
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                tokenize=True,
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if assistant_only_loss:
                                assistant_masks = prompt_completion_processed.get("assistant_masks")
                                if (
                                    assistant_masks is None
                                    or len(assistant_masks) != len(prompt_completion_ids)
                                    or 1 not in assistant_masks
                                ):
                                    raise ValueError(
                                        "assistant_only_loss=True but assistant masks are unavailable or invalid "
                                        "for a prompt-completion example (chat-template path)."
                                    )
                                output["assistant_masks"] = assistant_masks
                        else:
                            prompt_text = prompt_value if isinstance(prompt_value, str) else str(prompt_value)
                            completion_text = completion_value if isinstance(completion_value, str) else str(completion_value)
                            prompt_ids = processing_class(text=prompt_text)["input_ids"][1:]
                            prompt_completion_ids = processing_class(text=prompt_text + completion_text)[
                                "input_ids"
                            ][1:]

                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask
                    else:  # language modeling case
                        conversation = None
                        for conversation_key in (dataset_text_field, "messages", "conversations"):
                            if conversation_key in example and _looks_like_chat_messages(example[conversation_key]):
                                conversation = example[conversation_key]
                                break

                        if conversation is not None:
                            processed = processing_class.apply_chat_template(
                                conversation,
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                tokenize=True,
                                **example.get("chat_template_kwargs", {}),
                            )
                            output = {"input_ids": processed["input_ids"]}
                            if assistant_only_loss:
                                assistant_masks = processed.get("assistant_masks")
                                if (
                                    assistant_masks is None
                                    or len(assistant_masks) != len(processed["input_ids"])
                                    or 1 not in assistant_masks
                                ):
                                    raise ValueError(
                                        "assistant_only_loss=True but assistant masks are unavailable or invalid "
                                        "for a conversational example."
                                    )
                                output["assistant_masks"] = assistant_masks
                        else:
                            text_value = example.get(dataset_text_field)
                            if text_value is None:
                                raise KeyError(
                                    f"Missing dataset_text_field '{dataset_text_field}' for a non-conversational example. "
                                    f"Available keys: {list(example.keys())}"
                                )
                            if not isinstance(text_value, str):
                                raise ValueError(
                                    f"dataset_text_field '{dataset_text_field}' must be string for non-conversational "
                                    f"examples, got {type(text_value).__name__}."
                                )
                            output = {"input_ids": processing_class(text=text_value)["input_ids"]}
                    return output

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs
                )

                if args.assistant_only_loss:
                    if "assistant_masks" not in dataset.column_names:
                        raise ValueError(
                            "assistant_only_loss=True but tokenization did not produce an 'assistant_masks' column. "
                            "This dataset/template path is not compatible with assistant-only loss."
                        )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in dataset.column_names:
                    columns.append("completion_mask")
                if "assistant_masks" in dataset.column_names:
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware flash attention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                dataset = dataset.select_columns(
                    {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}.intersection(dataset.column_names)
                )

            dataset = self._filter_no_assistant_token_rows(
                dataset=dataset,
                processing_class=processing_class,
                args=args,
                dataset_name=dataset_name,
            )

        return dataset
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(
                    [
                        "label",
                        "label_ids",
                        "temp_labels",
                        "top_p_labels",
                        "top_k_labels",
                        "completion_mask",
                        "assistant_masks",
                    ]
                    + self.label_names
                )
            )

    def _decode_token(self, token_id: int) -> str:
        processing_class = getattr(self, "processing_class", None)
        if processing_class is None:
            return ""
        try:
            return processing_class.decode([token_id], skip_special_tokens=False)
        except Exception:
            return ""

    def _decode_ids(self, token_ids: list[int]) -> str:
        processing_class = getattr(self, "processing_class", None)
        if processing_class is None:
            return ""
        try:
            return processing_class.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return ""

    def _top_token_probs(self, probs: torch.Tensor, k: int) -> list[dict[str, Any]]:
        top_k = min(k, int(probs.size(-1)))
        values, ids = torch.topk(probs, k=top_k, dim=-1)
        result = []
        for prob, token_id in zip(values.tolist(), ids.tolist()):
            token_id = int(token_id)
            result.append({
                "token_id": token_id,
                "token": self._decode_token(token_id),
                "prob": float(prob),
            })
        return result

    def _build_temp_diag_payload(
        self,
        outputs,
        inputs,
    ) -> Optional[tuple[dict[str, Any], Optional[torch.Tensor]]]:
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        if labels is None or input_ids is None:
            return None
        if outputs.logits is None or outputs.temp_logits is None:
            return None

        with torch.no_grad():
            logits = outputs.logits.detach()
            temp_logits = outputs.temp_logits.detach()
            labels = labels.detach()
            input_ids = input_ids.detach()

            if logits.ndim != 3 or temp_logits.ndim != 3 or labels.ndim != 2:
                return None
            if logits.size(1) < 2 or labels.size(1) < 2:
                return None

            shift_logits = logits[:, :-1, :]
            shift_temps = temp_logits[:, :-1, :].clamp_min(1e-2).squeeze(-1)
            shift_labels = labels[:, 1:]

            model_valid_mask = getattr(outputs, "temp_training_valid_mask", None)
            if (
                isinstance(model_valid_mask, torch.Tensor)
                and model_valid_mask.ndim == 2
                and tuple(model_valid_mask.shape) == tuple(shift_labels.shape)
            ):
                valid_mask = model_valid_mask.detach().to(dtype=torch.bool, device=shift_labels.device)
            else:
                valid_mask = shift_labels != -100

            valid_indices = torch.nonzero(valid_mask, as_tuple=False)
            if valid_indices.numel() == 0:
                return None

            min_p = float(min(max(self.min_p_ratio, 1e-6), 1.0 - 1e-6))
            denom = -math.log(min_p)
            logits_valid = shift_logits[valid_mask]
            labels_valid = shift_labels[valid_mask]
            temp_valid = shift_temps[valid_mask]

            selected_mask = None
            model_selected_mask = getattr(outputs, "temp_training_selected_mask", None)
            if isinstance(model_selected_mask, torch.Tensor) and model_selected_mask.ndim == 1:
                candidate_selected = model_selected_mask.detach().to(
                    dtype=torch.bool,
                    device=labels_valid.device,
                )
                if candidate_selected.numel() == labels_valid.numel():
                    selected_mask = candidate_selected

            if selected_mask is None:
                selected_mask = torch.ones_like(labels_valid, dtype=torch.bool)

            safe_labels = shift_labels.clamp_min(0)
            gt_logits_full = shift_logits.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
            max_logits_full = shift_logits.max(dim=-1).values
            required_full = torch.relu(max_logits_full - gt_logits_full) / denom

            def _smooth_required_temp(values: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
                if window <= 1:
                    return values
                smooth = values.clone()
                for batch_idx in range(values.size(0)):
                    valid = mask[batch_idx]
                    if not valid.any():
                        continue
                    seq_vals = values[batch_idx][valid]
                    if seq_vals.numel() < 2:
                        smooth[batch_idx][valid] = seq_vals
                        continue
                    effective_window = min(int(window), int(seq_vals.numel()))
                    if effective_window < 2:
                        smooth[batch_idx][valid] = seq_vals
                        continue
                    effective_pad = effective_window // 2
                    padded = F.pad(
                        seq_vals.view(1, 1, -1),
                        (effective_pad, effective_pad),
                        mode="reflect",
                    )
                    avg = F.avg_pool1d(
                        padded,
                        kernel_size=effective_window,
                        stride=1,
                    ).view(-1)
                    if avg.numel() > seq_vals.numel():
                        avg = avg[:seq_vals.numel()]
                    smooth[batch_idx][valid] = avg
                return smooth

            smooth_window = int(self.temp_target_smooth_window)
            if smooth_window > 1:
                required_full = _smooth_required_temp(required_full, valid_mask, smooth_window)

            required_temp = required_full[valid_mask]
            cap_value = float(self.goldilocks_temp_cap)
            if cap_value >= 0.0:
                temp_cap = torch.as_tensor(
                    self.goldilocks_temp_cap,
                    device=required_temp.device,
                    dtype=required_temp.dtype,
                ).clamp_min(1e-6)
                within_cap = required_temp <= temp_cap
                selected_mask = selected_mask & within_cap
                if not selected_mask.any():
                    return None
                required_temp = torch.minimum(required_temp, temp_cap)
            feasible_mask = required_temp <= 2.0
            selected_mask = selected_mask & feasible_mask
            if not selected_mask.any():
                return None
            selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
            if selected_indices.numel() == 0:
                return None

            sample_count = min(self.temp_diag_examples, int(selected_indices.numel()))
            perm = torch.randperm(selected_indices.numel(), device=selected_indices.device)
            sampled_indices = selected_indices[perm[:sample_count]].sort().values

            target_values = required_temp[selected_mask] if selected_mask is not None else required_temp
            hinge_gap = required_temp - temp_valid
            examples = []

            for flat_idx in sampled_indices.tolist():
                batch_idx = int(valid_indices[flat_idx, 0].item())
                token_pos = int(valid_indices[flat_idx, 1].item())
                next_token_id = int(labels_valid[flat_idx].item())
                logit_vec = logits_valid[flat_idx]
                pred_temp = float(temp_valid[flat_idx].item())
                req_temp = float(required_temp[flat_idx].item())
                gap = float(hinge_gap[flat_idx].item())

                probs_unscaled = torch.softmax(logit_vec, dim=-1)
                probs_scaled = torch.softmax(logit_vec / max(pred_temp, 1e-2), dim=-1)
                gt_prob_unscaled = float(probs_unscaled[next_token_id].item())
                gt_prob_scaled = float(probs_scaled[next_token_id].item())
                max_prob_scaled = float(probs_scaled.max().item())
                threshold = float(min_p * max_prob_scaled)
                min_p_ok = gt_prob_scaled >= threshold

                gt_rank = int((logit_vec > logit_vec[next_token_id]).sum().item()) + 1
                pred_token_id = int(torch.argmax(probs_scaled).item())

                context_end = token_pos + 1
                context_start = max(0, context_end - 64)
                context_ids = input_ids[batch_idx, context_start:context_end].tolist()
                context_ids = [int(x) for x in context_ids]

                examples.append({
                    "batch_index": int(batch_idx),
                    "token_position": int(token_pos),
                    "target_token_position": int(token_pos + 1),
                    "context_token_ids": context_ids,
                    "context_text": self._decode_ids(context_ids),
                    "ground_truth": {
                        "token_id": next_token_id,
                        "token": self._decode_token(next_token_id),
                        "rank_unscaled": gt_rank,
                        "prob_unscaled": gt_prob_unscaled,
                        "prob_at_pred_temp": gt_prob_scaled,
                    },
                    "prediction": {
                        "top_token_id_at_pred_temp": pred_token_id,
                        "top_token_at_pred_temp": self._decode_token(pred_token_id),
                        "predicted_temperature": pred_temp,
                    },
                    "min_p_alignment": {
                        "min_p_ratio": min_p,
                        "required_temperature": req_temp,
                        "hinge_gap_required_minus_pred": gap,
                        "max_prob_at_pred_temp": max_prob_scaled,
                        "threshold_prob_min_p_times_max": threshold,
                        "condition_satisfied": bool(min_p_ok),
                    },
                    "top_tokens_unscaled": self._top_token_probs(probs_unscaled, self.temp_diag_topk),
                    "top_tokens_at_pred_temp": self._top_token_probs(probs_scaled, self.temp_diag_topk),
                })

            if not examples:
                return None

            target_summary = None
            if target_values is not None and target_values.numel() > 0:
                target_cpu = target_values.detach().float().cpu()
                target_summary = {
                    "count": int(target_cpu.numel()),
                    "mean": float(target_cpu.mean().item()),
                    "std": float(target_cpu.std(unbiased=False).item()),
                    "min": float(target_cpu.min().item()),
                    "p10": float(torch.quantile(target_cpu, 0.1).item()),
                    "p50": float(torch.quantile(target_cpu, 0.5).item()),
                    "p90": float(torch.quantile(target_cpu, 0.9).item()),
                    "max": float(target_cpu.max().item()),
                    "goldilocks_temp_cap": float(self.goldilocks_temp_cap),
                    "goldilocks_uniform": bool(self.goldilocks_uniform),
                    "goldilocks_uniform_bins": int(self.goldilocks_uniform_bins),
                    "temp_target_smooth_window": int(self.temp_target_smooth_window),
                }

            payload = {
                "global_step": int(self.state.global_step),
                "temp_objective": self.temp_objective,
                "temp_hinge_weight": float(self.temp_hinge_weight),
                "temp_reg_weight": float(self.temp_reg_weight),
                "goldilocks_temp_cap": float(self.goldilocks_temp_cap),
                "goldilocks_uniform": bool(self.goldilocks_uniform),
                "goldilocks_uniform_bins": int(self.goldilocks_uniform_bins),
                "temp_target_smooth_window": int(self.temp_target_smooth_window),
                "valid_token_count": int(valid_indices.size(0)),
                "selected_token_count_for_temp_loss": int(selected_indices.numel()),
                "target_distribution_summary": target_summary,
                "examples": examples,
            }
            return payload, target_values

    def _maybe_write_temp_diag(self, outputs, inputs) -> None:
        if not self.temp_diag_enabled:
            return
        if not self.is_world_process_zero():
            return
        if outputs.temp_logits is None:
            return

        step = int(self.state.global_step)
        if self._last_temp_diag_step == step:
            return
        if step % self.temp_diag_steps != 0:
            return

        result = self._build_temp_diag_payload(outputs, inputs)
        self._last_temp_diag_step = step
        if result is None:
            return
        payload, target_values = result

        os.makedirs(self._temp_diag_output_dir, exist_ok=True)
        out_path = os.path.join(self._temp_diag_output_dir, f"step_{step:07d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._write_temp_target_plot(step, target_values)
        print(f"[!] Wrote temp diagnostics: {out_path} ({len(payload['examples'])} examples)")

    def _write_temp_target_plot(self, step: int, target_values: Optional[torch.Tensor]) -> None:
        if target_values is None or target_values.numel() == 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return

        values = target_values.detach().float().cpu().numpy()
        if values.size < 2:
            return

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        cap_val = float(self.goldilocks_temp_cap)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        ax.hist(
            values,
            bins=60,
            density=True,
            color="#4C72B0",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.axvline(mean_val, color="#DD8452", linewidth=1.6, label=f"mean {mean_val:.3f}")
        ax.axvline(median_val, color="#55A868", linewidth=1.6, label=f"median {median_val:.3f}")
        if cap_val >= 0.0:
            ax.axvline(cap_val, color="#C44E52", linewidth=1.4, linestyle="--", label=f"cap {cap_val:.2f}")

        ax.set_title(f"Target Temperature Distribution (step {step})")
        ax.set_xlabel("Target temperature used for loss")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        out_path = os.path.join(self._temp_diag_output_dir, f"step_{step:07d}_target_dist.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs['temp_loss_weight'] = self.temp_loss_weight
        inputs["temp_objective"] = self.temp_objective
        inputs["min_p_ratio"] = self.min_p_ratio
        inputs["temp_hinge_weight"] = self.temp_hinge_weight
        inputs["temp_reg_weight"] = self.temp_reg_weight
        inputs["goldilocks_temp_cap"] = self.goldilocks_temp_cap
        inputs["goldilocks_uniform"] = self.goldilocks_uniform
        inputs["goldilocks_uniform_bins"] = self.goldilocks_uniform_bins
        inputs["temp_target_smooth_window"] = self.temp_target_smooth_window
        inputs["easy_token_drop_prob"] = self.easy_token_drop_prob
        inputs["top_p_loss_method"] = self.top_p_loss_method
        outputs = model(**inputs)
        self._maybe_write_temp_diag(outputs, inputs)
        temp_loss = outputs.temp_loss.item() if outputs.temp_loss is not None else 0
        lm_loss = outputs.lm_loss.item() if outputs.lm_loss is not None else 0
        top_p_loss = outputs.top_p_loss.item() if outputs.top_p_loss is not None else 0
        self.log({"loss": outputs.loss.item(), "temp_loss": temp_loss, "lm_loss": lm_loss, "top_p_loss": top_p_loss})
        if return_outputs:
            return outputs["loss"], outputs
        return outputs["loss"]
