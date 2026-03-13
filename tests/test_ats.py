from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

ROOT = Path(__file__).resolve().parents[1]
from model.ats_auto import ATSModelForCausalLM, LinearATSHead, MLPATSHead, TransformerATSHead
from trl_train import DataCollatorForLanguageModeling, _filter_split_for_positive_assistant_masks

ATS_HEAD_SPEC = importlib.util.spec_from_file_location(
    "simpledeco_ats_head_test_module",
    ROOT / "simpledeco_vllm" / "vllm" / "model_executor" / "models" / "ats_head.py",
)
assert ATS_HEAD_SPEC is not None and ATS_HEAD_SPEC.loader is not None
ATS_HEAD_MODULE = importlib.util.module_from_spec(ATS_HEAD_SPEC)
ATS_HEAD_SPEC.loader.exec_module(ATS_HEAD_MODULE)
build_vllm_ats_head = ATS_HEAD_MODULE.build_ats_head


def _write_tiny_base_model(model_dir: Path) -> None:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "foo": 6,
        "bar": 7,
        "assistant": 8,
        "user": 9,
    }
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<bos>' + message['role'] + ':' + message['content'] + '<eos>' }}"
        "{% endfor %}"
    )
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=vocab["<bos>"],
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def _build_generation_mask_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hi": 4,
        "ans": 5,
    }
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}{{ message['content'] }}{% endif %}"
        "{% endfor %}"
    )
    return tokenizer


def _ats_config_namespace(calibration_type: str, hidden_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        feature_key="hidden_states",
        max_temperature=10.0,
        normalize_logits=False,
        calibration_type=calibration_type,
        intermediate_size=32,
        num_attention_heads=2,
        attention_dropout=0.0,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


class _FakeDecoder(nn.Module):
    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class _FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = _FakeDecoder(self.embed_tokens)
        self.config = SimpleNamespace(output_hidden_states=False)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits, hidden_states=None)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {"args": args, **kwargs}


@pytest.mark.parametrize("calibration_type", ["linear", "mlp", "transformer"])
def test_ats_hf_head_roundtrip(calibration_type: str, tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    _write_tiny_base_model(base_dir)
    model = ATSModelForCausalLM.from_pretrained(
        str(base_dir),
        calibration_type=calibration_type,
        feature_key="hidden_states",
        loss_type="selective_smoothing",
    )
    input_ids = torch.tensor([[1, 4, 5, 2]])
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.calibrated_logits.shape == outputs.logits.shape
    llm_trainable = [name for name, param in model.named_parameters() if name.startswith("llm.") and param.requires_grad]
    ats_trainable = [name for name, param in model.named_parameters() if name.startswith("ats_head.") and param.requires_grad]
    assert not llm_trainable
    assert ats_trainable
    save_dir = tmp_path / f"ats-{calibration_type}"
    model.save_pretrained(save_dir)
    assert (save_dir / "config.json").exists()
    assert (save_dir / "ats_head.safetensors").exists()
    reloaded = ATSModelForCausalLM.from_pretrained(str(save_dir))
    reloaded_outputs = reloaded(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    assert reloaded_outputs.calibrated_logits.shape == outputs.calibrated_logits.shape
    for key, value in model.ats_head.state_dict().items():
        torch.testing.assert_close(reloaded.ats_head.state_dict()[key], value)


@pytest.mark.parametrize("calibration_type", ["linear", "mlp", "transformer"])
def test_ats_vllm_head_matches_hf_head(calibration_type: str) -> None:
    config = _ats_config_namespace(calibration_type, hidden_size=8)
    if calibration_type == "linear":
        hf_head = LinearATSHead(
            in_features=8,
            feature_key="hidden_states",
            max_temperature=10.0,
            normalize_logits=False,
        )
    elif calibration_type == "mlp":
        hf_head = MLPATSHead(
            in_features=8,
            intermediate_size=32,
            hidden_act="silu",
            feature_key="hidden_states",
            max_temperature=10.0,
            normalize_logits=False,
        )
    else:
        hf_head = TransformerATSHead(
            in_features=8,
            intermediate_size=32,
            num_attention_heads=2,
            attention_dropout=0.0,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
            attention_bias=False,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            feature_key="hidden_states",
            max_temperature=10.0,
            normalize_logits=False,
        )
    vllm_head = build_vllm_ats_head(config)
    vllm_head.load_state_dict(hf_head.state_dict(), strict=True)
    hidden_states = torch.randn(2, 5, 8)
    logits = torch.randn(2, 5, 10)
    lm_head_weight = torch.randn(10, 8)
    attention_mask = torch.ones(2, 5)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    hf_logits, hf_scale = hf_head(
        hidden_states=hidden_states,
        logits=logits,
        lm_head_weight=lm_head_weight,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    vllm_scale = vllm_head.get_temperature_scale(
        hidden_states=hidden_states,
        logits=logits,
        lm_head_weight=lm_head_weight,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    vllm_logits = vllm_head.apply_scale(logits, vllm_scale)
    torch.testing.assert_close(vllm_scale, hf_scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(vllm_logits, hf_logits, atol=1e-5, rtol=1e-5)


def test_ats_falls_back_to_decoder_hidden_states() -> None:
    config = ATSModelForCausalLM._to_ats_config(
        GPT2Config(
            vocab_size=11,
            n_positions=64,
            n_ctx=64,
            n_embd=8,
            n_layer=1,
            n_head=2,
        ),
        pretrained_model_name_or_path="dummy-base",
        calibration_type="linear",
        feature_key="hidden_states",
    )
    model = ATSModelForCausalLM(config, load_base_model=False)
    model.llm = _FakeCausalLM(vocab_size=11, hidden_size=8)
    model.llm.config.output_hidden_states = True
    input_ids = torch.tensor([[1, 2, 3]])
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        output_hidden_states=True,
    )
    assert outputs.loss is not None
    assert outputs.hidden_states is not None
    assert outputs.hidden_states[0].shape == (1, 3, 8)
    assert outputs.calibrated_logits.shape == outputs.logits.shape


def test_ats_collator_masks_assistant_tokens() -> None:
    collator = DataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=False,
        require_assistant_masks=True,
    )
    batch = collator(
        [
            {
                "input_ids": [1, 4, 5, 2],
                "assistant_masks": [0, 0, 1, 1],
            }
        ]
    )
    assert batch["labels"].tolist()[0][:2] == [-100, -100]
    assert batch["labels"].tolist()[0][2:] == [5, 2]


def test_ats_collator_allows_zero_assistant_mask_examples() -> None:
    collator = DataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=False,
        require_assistant_masks=True,
    )
    batch = collator(
        [
            {
                "input_ids": [1, 4, 5, 2],
                "assistant_masks": [0, 0, 0, 0],
            }
        ]
    )
    assert batch["labels"].tolist()[0] == [-100, -100, -100, -100]


def test_filter_split_for_positive_assistant_masks_removes_empty_rows() -> None:
    tokenizer = _build_generation_mask_tokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ans"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},
                ]
            },
        ]
    )
    filtered = _filter_split_for_positive_assistant_masks(
        dataset,
        tokenizer=tokenizer,
        text_field="messages",
        split_name="train",
    )
    assert len(filtered) == 1
    assert filtered[0]["messages"][-1]["content"] == "ans"


def test_trl_train_ats_smoke(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    _write_tiny_base_model(base_dir)
    dataset_path = tmp_path / "train.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "world"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "foo"},
                        {"role": "assistant", "content": "bar"},
                    ]
                },
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "trl_train.py"),
        "--model_name_or_path",
        str(base_dir),
        "--dataset_name",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "1e-3",
        "--max_steps",
        "1",
        "--logging_steps",
        "1",
        "--report_to",
        "none",
        "--train_ats",
        "true",
        "--gradient_checkpointing",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    assert (output_dir / "config.json").exists()
    assert (output_dir / "ats_head.safetensors").exists()
    assert not any(path.name.startswith("model") and path.suffix == ".safetensors" for path in output_dir.iterdir())
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "ats"
    head_state = load_file(str(output_dir / "ats_head.safetensors"))
    assert head_state
    assert all(key.startswith("ats_head.") for key in head_state)
