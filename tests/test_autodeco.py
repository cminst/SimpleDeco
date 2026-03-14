from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from model.templlm_auto import AutoDecoModelForCausalLM
from trainer.trl_autodeco import AutoDecoLLMTrainer


def _write_tiny_base_model(model_dir: Path) -> None:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
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


def test_autodeco_from_base_model_without_construct(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    _write_tiny_base_model(base_dir)

    model = AutoDecoModelForCausalLM.from_pretrained(str(base_dir))
    assert model.config.base_model_name_or_path == str(base_dir)
    assert model.llm is not None

    with torch.no_grad():
        for param in model.temp_head.parameters():
            param.fill_(0.25)
        for param in model.top_p_head.parameters():
            param.fill_(0.75)

    save_dir = tmp_path / "autodeco"
    model.save_pretrained(save_dir)

    reloaded = AutoDecoModelForCausalLM.from_pretrained(str(save_dir))
    assert reloaded.config.base_model_name_or_path == str(base_dir)
    for key, value in model.temp_head.state_dict().items():
        torch.testing.assert_close(reloaded.temp_head.state_dict()[key], value)
    for key, value in model.top_p_head.state_dict().items():
        torch.testing.assert_close(reloaded.top_p_head.state_dict()[key], value)


def test_autodeco_mean_head_prediction_uses_supervised_tokens_only() -> None:
    head_logits = torch.tensor([[[0.2], [0.4], [0.6], [0.8]]], dtype=torch.float32)
    labels = torch.tensor([[1, -100, 3, 4]], dtype=torch.long)
    mean_value = AutoDecoLLMTrainer._mean_head_prediction(head_logits, labels)
    assert mean_value is not None
    assert abs(mean_value - 0.5) < 1e-6
