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
            easy_token_drop_prob: float = 0.6,
            goldilocks_filter: bool = False,
            goldilocks_easy_frac: float = 0.1,
            goldilocks_topk_frac: float = 0.9,
            goldilocks_topk: int = 10,
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
        self.easy_token_drop_prob = easy_token_drop_prob
        self.goldilocks_filter = goldilocks_filter
        self.goldilocks_easy_frac = goldilocks_easy_frac
        self.goldilocks_topk_frac = goldilocks_topk_frac
        self.goldilocks_topk = goldilocks_topk
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

    def _render_row_with_bold_assistant_tokens(
        self,
        input_ids: list[int],
        assistant_masks: list[int],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
    ) -> str:
        pieces: list[str] = []
        for token_id, mask_value in zip(input_ids, assistant_masks):
            token_text = processing_class.decode([int(token_id)], skip_special_tokens=False)
            if int(mask_value) == 1:
                pieces.append(f"**{token_text}**")
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
            if self.is_world_process_zero():
                print(
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
                    print("[!] Example row with assistant tokens in bold:")
                    print(rendered)
            if total_after == 0:
                raise ValueError(
                    "assistant_only_loss=True but every example has zero assistant tokens after preprocessing."
                )
            return dataset

        dataset = dataset.filter(_has_assistant_tokens)
        if self.is_world_process_zero():
            print(
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
                            # prompt = [{"role": "user", "content": example["prompt"]}]
                            # response = [{"role": "assistant", "content": example["completion"]}]
                            # prompt_ids = processing_class.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
                            # response_ids = processing_class.apply_chat_template(response, tokenize=True, add_generation_prompt=False)[3:-1]
                            # prompt_completion_ids = prompt_ids + response_ids
                            
                            prompt_text = prompt_value if isinstance(prompt_value, str) else str(prompt_value)
                            completion_text = completion_value if isinstance(completion_value, str) else str(completion_value)
                            prompt_ids = processing_class(text=prompt_text)["input_ids"][1:]
                            prompt_completion_ids = processing_class(text=prompt_text + completion_text)[
                                "input_ids"
                            ][1:]
                
                            # prompt_completion_ids = example['token_ids']
                        # Check if the tokenized prompt starts with the tokenized prompt+completion
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
                    **map_kwargs,
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

    @staticmethod
    def _sample_indices(mask: torch.Tensor, target_count: int) -> torch.Tensor:
        if target_count <= 0:
            return torch.empty(0, dtype=torch.long, device=mask.device)
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() <= target_count:
            return indices
        perm = torch.randperm(indices.numel(), device=mask.device)
        return indices[perm[:target_count]]

    def _build_goldilocks_mask(
        self,
        logits_valid: torch.Tensor,
        labels_valid: torch.Tensor,
        easy_frac: float,
        topk_frac: float,
        topk: int,
    ) -> torch.Tensor:
        token_count = labels_valid.numel()
        if token_count == 0:
            return torch.zeros(0, dtype=torch.bool, device=labels_valid.device)

        k = max(1, min(int(topk), logits_valid.size(-1)))
        gt_logits = logits_valid.gather(1, labels_valid.unsqueeze(-1)).squeeze(-1)
        gt_rank = (logits_valid > gt_logits.unsqueeze(-1)).sum(dim=-1) + 1

        easy_mask = gt_rank == 1
        topk_non_easy_mask = (gt_rank <= k) & (~easy_mask)

        easy_count = int(easy_mask.sum().item())
        topk_count = int(topk_non_easy_mask.sum().item())
        available_total = easy_count + topk_count
        if available_total == 0:
            return torch.zeros(token_count, dtype=torch.bool, device=labels_valid.device)

        easy_raw = max(0.0, float(easy_frac))
        topk_raw = max(0.0, float(topk_frac))
        weight_sum = easy_raw + topk_raw
        if weight_sum <= 0.0:
            selected = torch.zeros(token_count, dtype=torch.bool, device=labels_valid.device)
            selected[easy_mask | topk_non_easy_mask] = True
            return selected

        easy_weight = easy_raw / weight_sum
        topk_weight = topk_raw / weight_sum

        if easy_count > 0 and topk_count > 0 and easy_weight > 0.0 and topk_weight > 0.0:
            max_total_from_easy = easy_count / easy_weight
            max_total_from_topk = topk_count / topk_weight
            target_total = int(min(max_total_from_easy, max_total_from_topk))
            target_total = max(1, min(available_total, target_total))
            target_easy = int(round(target_total * easy_weight))
            target_easy = max(0, min(easy_count, target_easy))
            target_topk = target_total - target_easy
            target_topk = max(0, min(topk_count, target_topk))

            remainder = target_total - (target_easy + target_topk)
            if remainder > 0:
                easy_spare = easy_count - target_easy
                take_easy = min(remainder, easy_spare)
                target_easy += take_easy
                remainder -= take_easy
            if remainder > 0:
                topk_spare = topk_count - target_topk
                take_topk = min(remainder, topk_spare)
                target_topk += take_topk
        elif topk_count > 0:
            target_easy = 0
            target_topk = topk_count
        else:
            target_easy = easy_count
            target_topk = 0

        selected = torch.zeros(token_count, dtype=torch.bool, device=labels_valid.device)
        if target_easy > 0:
            easy_idx = self._sample_indices(easy_mask, target_easy)
            selected[easy_idx] = True
        if target_topk > 0:
            topk_idx = self._sample_indices(topk_non_easy_mask, target_topk)
            selected[topk_idx] = True
        return selected

    def _build_temp_diag_payload(self, outputs, inputs) -> Optional[dict[str, Any]]:
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
                if self.temp_objective == "analytic_min_p_hinge" and self.goldilocks_filter:
                    selected_mask = self._build_goldilocks_mask(
                        logits_valid=logits_valid,
                        labels_valid=labels_valid,
                        easy_frac=self.goldilocks_easy_frac,
                        topk_frac=self.goldilocks_topk_frac,
                        topk=self.goldilocks_topk,
                    )

            selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
            if selected_indices.numel() == 0:
                return None

            sample_count = min(self.temp_diag_examples, int(selected_indices.numel()))
            perm = torch.randperm(selected_indices.numel(), device=selected_indices.device)
            sampled_indices = selected_indices[perm[:sample_count]].sort().values

            gt_logits = logits_valid.gather(1, labels_valid.unsqueeze(-1)).squeeze(-1)
            max_logits = logits_valid.max(dim=-1).values
            required_temp = torch.relu(max_logits - gt_logits) / denom
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

            payload = {
                "global_step": int(self.state.global_step),
                "temp_objective": self.temp_objective,
                "goldilocks_filter": bool(self.goldilocks_filter),
                "goldilocks_easy_frac": float(self.goldilocks_easy_frac),
                "goldilocks_topk_frac": float(self.goldilocks_topk_frac),
                "goldilocks_topk": int(self.goldilocks_topk),
                "temp_hinge_weight": float(self.temp_hinge_weight),
                "temp_reg_weight": float(self.temp_reg_weight),
                "valid_token_count": int(valid_indices.size(0)),
                "selected_token_count_for_temp_loss": int(selected_indices.numel()),
                "examples": examples,
            }
            return payload

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

        payload = self._build_temp_diag_payload(outputs, inputs)
        self._last_temp_diag_step = step
        if payload is None:
            return

        os.makedirs(self._temp_diag_output_dir, exist_ok=True)
        out_path = os.path.join(self._temp_diag_output_dir, f"step_{step:07d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[!] Wrote temp diagnostics: {out_path} ({len(payload['examples'])} examples)")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs['temp_loss_weight'] = self.temp_loss_weight
        inputs["temp_objective"] = self.temp_objective
        inputs["min_p_ratio"] = self.min_p_ratio
        inputs["temp_hinge_weight"] = self.temp_hinge_weight
        inputs["temp_reg_weight"] = self.temp_reg_weight
        inputs["easy_token_drop_prob"] = self.easy_token_drop_prob
        inputs["goldilocks_filter"] = self.goldilocks_filter
        inputs["goldilocks_easy_frac"] = self.goldilocks_easy_frac
        inputs["goldilocks_topk_frac"] = self.goldilocks_topk_frac
        inputs["goldilocks_topk"] = self.goldilocks_topk
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
