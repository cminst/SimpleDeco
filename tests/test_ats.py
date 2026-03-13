from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
from model.ats_auto import ATSModelForCausalLM, LinearATSHead, MLPATSHead, TransformerATSHead
from trl_train import DataCollatorForLanguageModeling

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
