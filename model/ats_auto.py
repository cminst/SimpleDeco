from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import SpecificPreTrainedModelType
from transformers.utils import ModelOutput, logging

try:
    from transformers import AutoModelForCausalLM as AutoModelForCausalLMClass
except ImportError:  # pragma: no cover
    AutoModelForCausalLMClass = AutoModelForCausalLM

logger = logging.get_logger(__name__)

ATS_HEAD_SAFETENSORS_NAME = "ats_head.safetensors"
ATS_HEAD_WEIGHTS_NAME = "ats_head.bin"


@dataclass
class ATSOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    calibrated_logits: Optional[torch.FloatTensor] = None
    ats_temperature_scale: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor, ...], ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def _feature_dim_for_key(feature_key: str, hidden_size: int) -> int:
    if feature_key in {"hidden_states", "output_token_feature"}:
        return hidden_size
    if feature_key in {"maxlogit", "logits_std"}:
        return 1
    raise ValueError(f"Unsupported ATS feature_key component: {feature_key}")


def _build_uniform_targets(logits: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    num_classes = logits.size(-1)
    targets = torch.full_like(logits, fill_value=label_smoothing / num_classes)
    return targets


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)


class TopKSmoothingLoss(nn.Module):
    def __init__(self, k: int = 5, label_smoothing: float = 0.1):
        super().__init__()
        self.k = k
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        targets = _build_uniform_targets(logits, self.label_smoothing)
        topk = logits.topk(k=min(self.k, logits.size(-1)), dim=-1).indices
        topk_probs = torch.full_like(topk, 1.0 / topk.size(-1), dtype=logits.dtype)
        targets.zero_().scatter_(-1, topk, topk_probs)
        hard_loss = F.cross_entropy(logits, labels, reduction="none") * (1.0 - self.label_smoothing)
        soft_loss = _soft_cross_entropy(logits, targets) * self.label_smoothing
        return hard_loss + soft_loss


class SelectiveSmoothingLoss(nn.Module):
    def __init__(
        self,
        label_smoothing_type: str = "uniform",
        smooth_loss_weight: float = 0.5,
        label_smoothing: float = 1.0,
        weighted_average: bool = True,
        k: int = 5,
    ):
        super().__init__()
        self.hard_loss = nn.CrossEntropyLoss(reduction="none")
        self.weighted_average = weighted_average
        self.smooth_loss_weight = smooth_loss_weight
        if label_smoothing_type == "uniform":
            self.smooth_loss = nn.CrossEntropyLoss(
                reduction="none",
                label_smoothing=label_smoothing,
            )
        elif label_smoothing_type == "topk":
            self.smooth_loss = TopKSmoothingLoss(
                k=k,
                label_smoothing=label_smoothing,
            )
        else:
            raise ValueError(
                "ATS only supports label_smoothing_type in {'uniform', 'topk'}."
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        correct_mask = logits.argmax(dim=-1) == labels
        hard_loss = (
            self.hard_loss(logits[correct_mask], labels[correct_mask])
            if correct_mask.any()
            else logits.new_zeros(1)
        )
        smooth_loss = (
            self.smooth_loss(logits[~correct_mask], labels[~correct_mask])
            if (~correct_mask).any()
            else logits.new_zeros(1)
        )
        if self.weighted_average:
            correct_ratio = correct_mask.float().mean()
            smooth_weight = self.smooth_loss_weight * correct_ratio
            hard_weight = (1.0 - self.smooth_loss_weight) * (1.0 - correct_ratio)
            total_weight = (smooth_weight + hard_weight).clamp_min(1e-8)
            hard_loss = hard_loss.mean() * (hard_weight / total_weight).to(hard_loss.device)
            smooth_loss = smooth_loss.mean() * (smooth_weight / total_weight).to(smooth_loss.device)
            return hard_loss + smooth_loss
        return (
            hard_loss.mean() * (1.0 - self.smooth_loss_weight)
            + smooth_loss.mean() * self.smooth_loss_weight
        )


def get_ats_loss_fn(config: "ATSConfig") -> nn.Module:
    if config.loss_type == "xent":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.loss_type == "topk_smoothing":
        return TopKSmoothingLoss(
            k=config.smoothing_topk,
            label_smoothing=config.label_smoothing,
        )
    if config.loss_type == "selective_smoothing":
        return SelectiveSmoothingLoss(
            label_smoothing_type=config.label_smoothing_type,
            smooth_loss_weight=config.smooth_loss_weight,
            label_smoothing=config.label_smoothing,
            k=config.smoothing_topk,
        )
    raise ValueError(f"Unsupported ATS loss_type: {config.loss_type}")


def get_feature(
    feature_key: str,
    *,
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    if feature_key == "hidden_states":
        return hidden_states
    if feature_key == "maxlogit":
        return logits.max(dim=-1, keepdim=True).values
    if feature_key == "output_token_feature":
        top_indices = logits.argmax(dim=-1)
        return lm_head_weight[top_indices]
    if feature_key == "logits_std":
        return torch.log(logits.std(dim=-1, keepdim=True).clamp_min(1e-8))
    raise ValueError(f"Unsupported ATS feature_key: {feature_key}")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float()
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_key_value_heads: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=torch.finfo(attn_weights.dtype).min,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class ATSMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
        )
        self.mlp = ATSMLP(hidden_size, intermediate_size, hidden_act=hidden_act)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class BaseATSHead(nn.Module):
    def __init__(
        self,
        *,
        feature_key: str,
        max_temperature: float,
        normalize_logits: bool,
        prediction_type: str = "temperature",
    ):
        super().__init__()
        self.feature_keys = feature_key.split("+")
        self.max_temperature = max_temperature
        self.normalize_logits = normalize_logits
        self.prediction_type = prediction_type

    def construct_features(
        self,
        *,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        features = [
            get_feature(
                key,
                hidden_states=hidden_states,
                logits=logits,
                lm_head_weight=lm_head_weight,
            )
            for key in self.feature_keys
        ]
        return torch.cat(features, dim=-1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        lm_head_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.construct_features(
            hidden_states=hidden_states,
            logits=logits,
            lm_head_weight=lm_head_weight,
        )
        head_output = self.get_head_output(
            features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        scaled_logits = logits
        if self.normalize_logits:
            scaled_logits = scaled_logits / scaled_logits.std(dim=-1, keepdim=True).clamp_min(1e-8)
        temperature_scale = torch.exp(head_output).clamp(max=self.max_temperature)
        return scaled_logits * temperature_scale, temperature_scale


class LinearATSHead(BaseATSHead):
    def __init__(self, in_features: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        return self.linear(features.to(self.linear.weight.dtype))


class MLPATSHead(BaseATSHead):
    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.mlp = ATSMLP(in_features, intermediate_size, hidden_act=hidden_act)
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        features = features.to(self.linear.weight.dtype)
        return self.linear(self.mlp(features))


class TransformerATSHead(BaseATSHead):
    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        num_key_value_heads: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.transformer = DecoderLayer(
            hidden_size=in_features,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
        )
        self.linear = nn.Linear(in_features, 1)

    def get_head_output(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        features = features.to(self.linear.weight.dtype)
        hidden_states = self.transformer(
            hidden_states=features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.linear(hidden_states)


class ATSConfig(PretrainedConfig):
    model_type = "ats"
    has_no_defaults_at_init = True

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        base_model_type: Optional[str] = None,
        calibration_type: str = "transformer",
        feature_key: str = "hidden_states",
        freeze_base_model: bool = True,
        normalize_logits: bool = False,
        max_temperature: float = 10.0,
        loss_type: str = "selective_smoothing",
        label_smoothing: float = 1.0,
        smooth_loss_weight: float = 0.5,
        label_smoothing_type: str = "uniform",
        smoothing_topk: int = 5,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        overwrite_logits: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        hidden_size = kwargs.get("hidden_size") or kwargs.get("n_embd")
        if hidden_size is None:
            raise ValueError("ATSConfig requires hidden_size from the base model config.")
        num_attention_heads = num_attention_heads or kwargs.get("num_attention_heads") or kwargs.get("n_head")
        max_position_embeddings = kwargs.get("max_position_embeddings") or kwargs.get("n_positions") or kwargs.get("n_ctx") or max_position_embeddings
        self.base_model_name_or_path = base_model_name_or_path
        if base_model_name_or_path is not None:
            self._name_or_path = base_model_name_or_path
        self.base_model_type = base_model_type or kwargs.get("base_model_type") or kwargs.get("model_type")
        self.hidden_size = hidden_size
        self.calibration_type = calibration_type
        self.feature_key = feature_key
        self.freeze_base_model = freeze_base_model
        self.normalize_logits = normalize_logits
        self.max_temperature = max_temperature
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.smooth_loss_weight = smooth_loss_weight
        self.label_smoothing_type = label_smoothing_type
        self.smoothing_topk = smoothing_topk
        self.in_features = hidden_size
        self.intermediate_size = intermediate_size or kwargs.get(
            "intermediate_size",
            hidden_size * 4,
        )
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or kwargs.get(
            "num_key_value_heads",
            self.num_attention_heads,
        )
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.overwrite_logits = overwrite_logits
        self.architectures = ["ATSModelForCausalLM"]


class ATSModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ATSConfig
    supports_gradient_checkpointing = True
    _no_split_modules: list[str] = []

    def __init__(self, config: ATSConfig, **kwargs: Any):
        super().__init__(config)
        if config.base_model_name_or_path is None:
            raise ValueError("ATSConfig.base_model_name_or_path must be set.")
        self.config = config
        self.overwrite_logits = config.overwrite_logits
        if kwargs.get("load_base_model") is False:
            self.llm = None
        else:
            base_model_kwargs = kwargs.get("base_model_kwargs", {})
            base_model_args = kwargs.get("base_model_args", ())
            if "dtype" in base_model_kwargs and "torch_dtype" not in base_model_kwargs:
                base_model_kwargs["torch_dtype"] = base_model_kwargs.pop("dtype")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                *base_model_args,
                **base_model_kwargs,
            )
        self.ats_head = self._build_head()
        self.loss_fn = get_ats_loss_fn(config)
        if self.llm is not None and config.freeze_base_model:
            for param in self.llm.parameters():
                param.requires_grad = False
        if self.llm is not None and hasattr(self.llm.config, "attn_implementation"):
            self.llm.config.attn_implementation = getattr(
                config,
                "attn_implementation",
                getattr(self.llm.config, "attn_implementation", None),
            )
        if self.llm is not None and hasattr(self.llm, "_no_split_modules"):
            self._no_split_modules = self.llm._no_split_modules
        self._keys_to_ignore_on_save = [k for k in self.state_dict() if k.startswith("llm.")]

    def _head_input_dim(self) -> int:
        return sum(_feature_dim_for_key(key, self.config.hidden_size) for key in self.config.feature_key.split("+"))

    def _build_head(self) -> BaseATSHead:
        common_kwargs = {
            "feature_key": self.config.feature_key,
            "max_temperature": self.config.max_temperature,
            "normalize_logits": self.config.normalize_logits,
        }
        in_features = self._head_input_dim()
        if self.config.calibration_type == "linear":
            return LinearATSHead(in_features=in_features, **common_kwargs)
        if self.config.calibration_type == "mlp":
            return MLPATSHead(
                in_features=in_features,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
                **common_kwargs,
            )
        if self.config.calibration_type == "transformer":
            return TransformerATSHead(
                in_features=in_features,
                intermediate_size=self.config.intermediate_size,
                num_attention_heads=self.config.num_attention_heads,
                attention_dropout=self.config.attention_dropout,
                num_key_value_heads=self.config.num_key_value_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                rope_theta=self.config.rope_theta,
                attention_bias=self.config.attention_bias,
                hidden_act=self.config.hidden_act,
                rms_norm_eps=self.config.rms_norm_eps,
                **common_kwargs,
            )
        raise ValueError(f"Unsupported ATS calibration_type: {self.config.calibration_type}")

    @staticmethod
    def _flatten_head_state_dict(head_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {f"ats_head.{key}": value for key, value in head_state_dict.items()}

    @staticmethod
    def _split_head_state_dict(flat_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key[len("ats_head."):]: value
            for key, value in flat_state_dict.items()
            if key.startswith("ats_head.")
        }

    @classmethod
    def _to_ats_config(
        cls,
        config: PretrainedConfig,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **overrides: Any,
    ) -> ATSConfig:
        if isinstance(config, ATSConfig):
            ats_config = config
        else:
            config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            base_model_type = config_dict.get("model_type", getattr(config, "model_type", None))
            ats_config = ATSConfig(
                **config_dict,
                base_model_name_or_path=str(pretrained_model_name_or_path),
                base_model_type=base_model_type,
            )
        if ats_config.base_model_name_or_path is None:
            ats_config.base_model_name_or_path = str(pretrained_model_name_or_path)
            ats_config._name_or_path = str(pretrained_model_name_or_path)
        for key, value in overrides.items():
            if value is not None and hasattr(ats_config, key):
                setattr(ats_config, key, value)
        return ats_config

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args: Any,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        **kwargs: Any,
    ) -> "ATSModelForCausalLM":
        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path must be provided")
        config_load_keys = (
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
            "use_auth_token",
        )
        config_kwargs = {key: kwargs[key] for key in config_load_keys if key in kwargs}
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        config_override_keys = {
            "calibration_type",
            "feature_key",
            "freeze_base_model",
            "normalize_logits",
            "max_temperature",
            "loss_type",
            "label_smoothing",
            "smooth_loss_weight",
            "label_smoothing_type",
            "smoothing_topk",
            "intermediate_size",
            "max_position_embeddings",
            "attention_dropout",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_act",
            "rope_theta",
            "attention_bias",
            "rms_norm_eps",
            "overwrite_logits",
        }
        config_overrides = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in config_override_keys}
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        is_ats_checkpoint = getattr(config, "model_type", None) == "ats"
        ats_config = cls._to_ats_config(
            config=config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **config_overrides,
        )
        if not is_ats_checkpoint:
            logger.info("Creating a new ATS model from base checkpoint: %s", pretrained_model_name_or_path)
            return cls(
                ats_config,
                base_model_args=model_args,
                base_model_kwargs=kwargs,
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            ats_config.base_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model = cls(ats_config, load_base_model=False)
        model.llm = base_model
        if hasattr(base_model, "_no_split_modules"):
            model._no_split_modules = base_model._no_split_modules
        checkpoint_path = Path(pretrained_model_name_or_path)
        safetensors_path = checkpoint_path / ATS_HEAD_SAFETENSORS_NAME
        pytorch_path = checkpoint_path / ATS_HEAD_WEIGHTS_NAME
        loaded = False
        if safetensors_path.exists():
            head_state = cls._split_head_state_dict(load_file(str(safetensors_path)))
            model.ats_head.load_state_dict(head_state, strict=True)
            loaded = True
        elif pytorch_path.exists():
            head_state = torch.load(pytorch_path, map_location="cpu")
            if "ats_head" in head_state:
                head_state = head_state["ats_head"]
            else:
                head_state = cls._split_head_state_dict(head_state)
            model.ats_head.load_state_dict(head_state, strict=True)
            loaded = True
        if not loaded:
            raise FileNotFoundError(
                f"Could not find ATS head weights at {checkpoint_path}. "
                f"Expected {ATS_HEAD_SAFETENSORS_NAME} or {ATS_HEAD_WEIGHTS_NAME}."
            )
        return model

    def set_overwrite_logits(self, set_to: bool = True) -> None:
        self.overwrite_logits = set_to
        self.config.overwrite_logits = set_to

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any):
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)

    def _filtered_llm_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(self.llm.forward)
        except (TypeError, ValueError):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def _compute_loss(
        self,
        calibrated_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = calibrated_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            return calibrated_logits.new_zeros(())
        flat_logits = shift_logits[valid_mask]
        flat_labels = shift_labels[valid_mask].to(flat_logits.device)
        return self.loss_fn(flat_logits, flat_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> ATSOutputWithPast:
        llm_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
        )
        llm_kwargs.update(kwargs)
        outputs = self.llm(**self._filtered_llm_kwargs(llm_kwargs))
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits if getattr(outputs, "logits", None) is not None else self.llm.lm_head(hidden_states)
        if position_ids is None:
            seq_len = hidden_states.size(1)
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            if input_ids is not None and input_ids.size(0) > 1:
                position_ids = position_ids.expand(input_ids.size(0), -1)
        calibrated_logits, temperature_scale = self.ats_head(
            hidden_states=hidden_states,
            logits=logits,
            lm_head_weight=self.llm.get_output_embeddings().weight,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        loss = self._compute_loss(calibrated_logits, labels) if labels is not None else None
        final_logits = calibrated_logits if self.overwrite_logits else logits
        hidden_states_to_return = outputs.hidden_states if output_hidden_states else None
        return ATSOutputWithPast(
            loss=loss,
            logits=final_logits,
            calibrated_logits=calibrated_logits,
            ats_temperature_scale=temperature_scale,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=hidden_states_to_return,
            attentions=getattr(outputs, "attentions", None),
        )

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        save_function: Any = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs: Any,
    ) -> None:
        del push_to_hub, variant, token, save_peft_format, kwargs
        from transformers.modeling_utils import unwrap_model

        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file.")
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = unwrap_model(self)
        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)
        fallback_state_dict = self._flatten_head_state_dict(model_to_save.ats_head.state_dict())
        if state_dict is None:
            state_dict = fallback_state_dict
        else:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if key.startswith("ats_head.")
            }
            if not state_dict:
                state_dict = fallback_state_dict
        weights_name = ATS_HEAD_SAFETENSORS_NAME if safe_serialization else ATS_HEAD_WEIGHTS_NAME
        filename_pattern = (
            weights_name.replace(".bin", "{suffix}.bin")
            .replace(".safetensors", "{suffix}.safetensors")
        )
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
            max_shard_size=max_shard_size,
        )
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
        for shard_file, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            if safe_serialization:
                save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))
        if index is not None:
            index_name = (
                "ats_head.safetensors.index.json"
                if safe_serialization
                else "ats_head.bin.index.json"
            )
            with open(os.path.join(save_directory, index_name), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(index, indent=2, sort_keys=True) + "\n")


try:
    AutoConfig.register("ats", ATSConfig)
except ValueError:
    pass
try:
    AutoModel.register(ATSConfig, ATSModelForCausalLM)
except ValueError:
    pass
try:
    AutoModelForCausalLMClass.register(ATSConfig, ATSModelForCausalLM)
except ValueError:
    pass


__all__ = [
    "ATSConfig",
    "ATSModelForCausalLM",
    "ATSOutputWithPast",
    "LinearATSHead",
    "MLPATSHead",
    "TransformerATSHead",
]
