#!/usr/bin/env python3
"""
Train and evaluate a logits-only shadow controller that imitates AutoDeco's
*induced next-token distribution* rather than only its raw (T_hat, p_hat).

The script is designed to be close to the user's existing workflow:
- loads a Hugging Face dataset from disk
- uses per-token diagnostics exported from AutoDeco
- trains a family of logits-only controllers with progressively richer inputs
- logs whether pre-registered hypotheses are supported or not

Core idea:
    q_auto  = HardTopP( softmax(logits / T_hat), p_hat )
    q_pred  = HardTopP( softmax(logits / T_pred), p_pred )

Training uses a differentiable soft-top-p surrogate for q_pred, but validation
and reported metrics always use hard top-p.

Usage example
-------------
python3 script/shadow_controller_dist_match.py \
    --path ckpt/pertoken_diagnostics/autodeco_qwen7b_dolci_val_balanced/ \
    --profile-k 64 \
    --dist-k 200 \
    --epochs 8 \
    --batch-size 8192 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("shadow_controller")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def quantize_np(x: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return x
    return np.round(x / step) * step


def r2_np(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def safe_div(n: float, d: float, eps: float = 1e-12) -> float:
    return float(n / (d + eps))


def batched_indices(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.RandomState) -> Iterable[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


MODEL_LABELS = {
    "mean": "Mean",
    "scalar": "Scalar",
    "scalar_bits": "Scalar+bits",
    "scalar_bits_topk": "Scalar+bits+topk",
}


LATEX_METRIC_ORDER = (
    ("js", "min"),
    ("entropy_mae", "min"),
    ("nucleus_mae", "min"),
    ("top1_mae", "min"),
    ("T_mae", "min"),
    ("p_mae", "min"),
    ("entropy_r2", "max"),
)


def _format_latex_metric(value: float, *, bold: bool = False, digits: int = 3) -> str:
    body = f"{value:.{digits}f}"
    if bold:
        return rf"$\mathbf{{{body}}}$"
    return f"${body}$"


def build_shadow_latex_rows(summary: Dict[str, Dict[str, float]]) -> List[str]:
    best_by_metric: Dict[str, float] = {}
    for metric_name, direction in LATEX_METRIC_ORDER:
        values = [float(summary[name][metric_name]) for name in MODEL_LABELS if name in summary]
        best_by_metric[metric_name] = min(values) if direction == "min" else max(values)

    rows: List[str] = []
    for name, label in MODEL_LABELS.items():
        metrics = summary[name]
        formatted_metrics = []
        for metric_name, direction in LATEX_METRIC_ORDER:
            value = float(metrics[metric_name])
            best_value = best_by_metric[metric_name]
            is_best = bool(np.isclose(value, best_value, rtol=0.0, atol=5e-7))
            formatted_metrics.append(_format_latex_metric(value, bold=is_best))
        rows.append(f"{label} & " + " & ".join(formatted_metrics) + r" \\")
    return rows


# -----------------------------------------------------------------------------
# Distribution helpers
# -----------------------------------------------------------------------------

def maybe_sort_desc_numpy(logits: np.ndarray, sample_rows: int = 4096) -> Tuple[np.ndarray, bool]:
    """Sort rows descending only if they appear unsorted on a small sample."""
    n = logits.shape[0]
    m = min(n, sample_rows)
    sample = logits[:m]
    unsorted_frac = float(np.mean(np.any(sample[:, 1:] > sample[:, :-1], axis=1)))
    if unsorted_frac > 0.0:
        logits = np.sort(logits, axis=1)[:, ::-1]
        return logits, True
    return logits, False


def hard_top_p_from_sorted_logits_torch(
    sorted_logits: torch.Tensor,
    T: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute hard top-p distribution from row-wise sorted logits.
    Args:
        sorted_logits: [B, K], descending order.
        T: [B]
        p: [B]
    Returns:
        q: [B, K] hard-top-p distribution on the K provided logits.
        stats: entropy, top1, nucleus_size, retained_mass_pre_renorm
    """
    T = T.clamp_min(1e-4)
    p = p.clamp(1e-6, 1.0)

    probs = torch.softmax(sorted_logits / T.unsqueeze(-1), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    first_ge = (cdf >= p.unsqueeze(-1)).float().argmax(dim=-1)
    ar = torch.arange(sorted_logits.shape[1], device=sorted_logits.device).unsqueeze(0)
    keep = ar <= first_ge.unsqueeze(-1)
    masked = probs * keep.float()
    retained = masked.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = masked / retained

    entropy = -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(dim=-1)
    top1 = q[:, 0]
    nucleus_size = keep.sum(dim=-1).float()
    return q, {
        "entropy": entropy,
        "top1": top1,
        "nucleus_size": nucleus_size,
        "retained_mass": retained.squeeze(-1),
    }


def soft_top_p_from_sorted_logits_torch(
    sorted_logits: torch.Tensor,
    T: torch.Tensor,
    p: torch.Tensor,
    soft_tau: float = 0.02,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Differentiable top-p surrogate.
    For token i, use cdf_prev(i) = sum_{j < i} prob_j and set a soft keep mask
        m_i = sigmoid((p - cdf_prev(i)) / soft_tau)
    Then renormalize the masked distribution.
    """
    T = T.clamp_min(1e-4)
    p = p.clamp(1e-6, 1.0)

    probs = torch.softmax(sorted_logits / T.unsqueeze(-1), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    cdf_prev = cdf - probs
    mask = torch.sigmoid((p.unsqueeze(-1) - cdf_prev) / soft_tau)
    masked = probs * mask
    retained = masked.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = masked / retained

    entropy = -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(dim=-1)
    top1 = q[:, 0]
    nucleus_size_soft = mask.sum(dim=-1)
    return q, {
        "entropy": entropy,
        "top1": top1,
        "nucleus_size": nucleus_size_soft,
        "retained_mass": retained.squeeze(-1),
    }


def batch_forward_kl(target_q: torch.Tensor, pred_q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean KL(target || pred) over batch."""
    t = target_q.clamp_min(eps)
    p = pred_q.clamp_min(eps)
    kl = (t * (t.log() - p.log())).sum(dim=-1)
    return kl.mean()


def batch_js_div(target_q: torch.Tensor, pred_q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (target_q + pred_q)
    return 0.5 * batch_forward_kl(target_q, m, eps=eps) + 0.5 * batch_forward_kl(pred_q, m, eps=eps)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class MeanController(nn.Module):
    def __init__(self, tmin: float, tmax: float, pmin: float, pmax: float):
        super().__init__()
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.uT = nn.Parameter(torch.zeros(()))
        self.up = nn.Parameter(torch.zeros(()))

    def forward(self, x: Optional[torch.Tensor], n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.tmin + (self.tmax - self.tmin) * torch.sigmoid(self.uT)
        p = self.pmin + (self.pmax - self.pmin) * torch.sigmoid(self.up)
        return T.expand(n), p.expand(n)


class MLPController(nn.Module):
    def __init__(self, d_in: int, hidden: int, tmin: float, tmax: float, pmin: float, pmax: float, dropout: float = 0.0):
        super().__init__()
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        uT = out[:, 0]
        up = out[:, 1]
        T = self.tmin + (self.tmax - self.tmin) * torch.sigmoid(uT)
        p = self.pmin + (self.pmax - self.pmin) * torch.sigmoid(up)
        return T, p


# -----------------------------------------------------------------------------
# Data loading / feature construction
# -----------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    X_mean: Optional[np.ndarray]
    X_scalar: np.ndarray
    X_scalar_bits: np.ndarray
    X_scalar_bits_topk: np.ndarray
    yT: np.ndarray
    yp: np.ndarray
    seq_id: np.ndarray
    topk_logits: np.ndarray
    mass_ref: Optional[np.ndarray]
    feature_names: Dict[str, List[str]]


@dataclass
class PreparedFeatureSet:
    name: str
    X: Optional[np.ndarray]
    mu: np.ndarray
    sd: np.ndarray
    n_binary: int
    feature_names: List[str]


FEATURE_SET_N_BINARY = {
    "mean": 0,
    "scalar": 0,
    "scalar_bits": 5,
    "scalar_bits_topk": 5,
}


def _as_2d_float32(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.float32)
    return np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)


def build_feature_bundle_from_token_data(
    data: Dict[str, np.ndarray],
    profile_k: int,
    dist_k: int,
    logger: logging.Logger,
) -> FeatureBundle:
    seq_id = data["seq_id"].astype(np.int64)
    yT = data["T_hat"].astype(np.float32)
    yp = data["p_hat"].astype(np.float32)

    # Continuous features.
    x_H = data["H_norm"].astype(np.float32)
    x_gap = np.log(data["gap12"].astype(np.float32) + 1e-6)
    x_pmax = data["p_max"].astype(np.float32)
    x_mass10 = data["mass10"].astype(np.float32)
    x_mass50 = data["mass50"].astype(np.float32)
    x_mass200 = data["mass200"].astype(np.float32)
    x_expH = data["expH"].astype(np.float32)
    x_pos = np.log1p(data["t"].astype(np.float32))

    # Binary / structural features.
    x_boundary = data["is_boundary"].astype(np.float32)
    x_punct = data["is_punct"].astype(np.float32)
    x_ws = data["is_whitespace"].astype(np.float32)
    x_nl = data["is_newline"].astype(np.float32)
    x_code = data["in_code_block"].astype(np.float32)

    topk_logits = _as_2d_float32(data["topk_logits"])
    if dist_k > topk_logits.shape[1]:
        raise ValueError(f"Requested dist_k={dist_k}, but dataset only has {topk_logits.shape[1]} logits per token.")
    if profile_k > dist_k:
        raise ValueError("profile_k must be <= dist_k")

    topk_logits = topk_logits[:, :dist_k]
    topk_logits, sorted_flag = maybe_sort_desc_numpy(topk_logits)
    if sorted_flag:
        logger.info("Detected unsorted top-k logits; sorted them descending once during preprocessing.")
    else:
        logger.info("Top-k logits appear pre-sorted descending.")

    # Top-k profile features are translation-invariant shifted logits.
    topk_profile = topk_logits[:, :profile_k] - topk_logits[:, :1]

    scalar_feats = np.stack(
        [x_H, x_gap, x_pmax, x_mass10, x_mass50, x_mass200, x_expH, x_pos],
        axis=1,
    ).astype(np.float32)
    bits = np.stack([x_boundary, x_punct, x_ws, x_nl, x_code], axis=1).astype(np.float32)

    X_scalar = scalar_feats
    X_scalar_bits = np.concatenate([scalar_feats, bits], axis=1).astype(np.float32)
    X_scalar_bits_topk = np.concatenate([scalar_feats, bits, topk_profile], axis=1).astype(np.float32)

    feature_names = {
        "scalar": ["H_norm", "log_gap12", "p_max", "mass10", "mass50", "mass200", "expH", "log1p_t"],
        "scalar_bits": ["H_norm", "log_gap12", "p_max", "mass10", "mass50", "mass200", "expH", "log1p_t",
                         "is_boundary", "is_punct", "is_whitespace", "is_newline", "in_code_block"],
        "scalar_bits_topk": ["H_norm", "log_gap12", "p_max", "mass10", "mass50", "mass200", "expH", "log1p_t",
                              "is_boundary", "is_punct", "is_whitespace", "is_newline", "in_code_block"]
                             + [f"shifted_logit_{i}" for i in range(profile_k)],
    }

    mass_ref = None
    if dist_k >= 200:
        mass_ref = x_mass200
    elif dist_k >= 50:
        mass_ref = x_mass50
    elif dist_k >= 10:
        mass_ref = x_mass10

    return FeatureBundle(
        X_mean=None,
        X_scalar=X_scalar,
        X_scalar_bits=X_scalar_bits,
        X_scalar_bits_topk=X_scalar_bits_topk,
        yT=yT,
        yp=yp,
        seq_id=seq_id,
        topk_logits=topk_logits,
        mass_ref=mass_ref,
        feature_names=feature_names,
    )


def build_feature_bundle(ds_path: str, profile_k: int, dist_k: int, logger: logging.Logger) -> FeatureBundle:
    ds = load_from_disk(ds_path)
    tok = ds["tokens"]

    cols = [
        "seq_id", "T_hat", "p_hat",
        "H_norm", "gap12", "p_max", "mass10", "mass50", "mass200", "expH", "t",
        "is_boundary", "is_punct", "is_whitespace", "is_newline", "in_code_block",
        "topk_logits",
    ]
    missing = [c for c in cols if c not in tok.features]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    tok = tok.select_columns(cols).with_format("numpy")
    data = tok[:]
    return build_feature_bundle_from_token_data(data, profile_k=profile_k, dist_k=dist_k, logger=logger)


def split_by_seq_mod(seq_id: np.ndarray, val_mod: int) -> Tuple[np.ndarray, np.ndarray]:
    val_mask = (seq_id % val_mod) == 0
    tr_mask = ~val_mask
    return tr_mask, val_mask


def standardize_inplace(X: np.ndarray, tr_mask: np.ndarray, n_binary: int) -> Tuple[np.ndarray, np.ndarray]:
    X = X.copy()
    n_cont = X.shape[1] - n_binary
    if n_cont > 0:
        mu = X[tr_mask, :n_cont].mean(axis=0)
        sd = X[tr_mask, :n_cont].std(axis=0) + 1e-6
        X[:, :n_cont] = (X[:, :n_cont] - mu) / sd
    else:
        mu = np.zeros((0,), dtype=np.float32)
        sd = np.ones((0,), dtype=np.float32)
    return mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, n_binary: int) -> np.ndarray:
    X = X.copy()
    n_cont = X.shape[1] - n_binary
    if n_cont > 0:
        X[:, :n_cont] = (X[:, :n_cont] - mu) / sd
    return X


def get_feature_matrix(bundle: FeatureBundle, feature_set: str) -> Optional[np.ndarray]:
    if feature_set == "mean":
        return None
    if feature_set == "scalar":
        return bundle.X_scalar
    if feature_set == "scalar_bits":
        return bundle.X_scalar_bits
    if feature_set == "scalar_bits_topk":
        return bundle.X_scalar_bits_topk
    raise ValueError(f"Unknown feature_set '{feature_set}'.")


def prepare_feature_sets(bundle: FeatureBundle, tr_mask: np.ndarray) -> Dict[str, PreparedFeatureSet]:
    prepared: Dict[str, PreparedFeatureSet] = {
        "mean": PreparedFeatureSet(
            name="mean",
            X=None,
            mu=np.zeros((0,), dtype=np.float32),
            sd=np.ones((0,), dtype=np.float32),
            n_binary=0,
            feature_names=[],
        )
    }

    for name in ("scalar", "scalar_bits", "scalar_bits_topk"):
        X_raw = get_feature_matrix(bundle, name)
        assert X_raw is not None
        n_binary = FEATURE_SET_N_BINARY[name]
        mu, sd = standardize_inplace(X_raw, tr_mask, n_binary=n_binary)
        prepared[name] = PreparedFeatureSet(
            name=name,
            X=apply_standardize(X_raw, mu, sd, n_binary=n_binary),
            mu=mu.astype(np.float32),
            sd=sd.astype(np.float32),
            n_binary=n_binary,
            feature_names=list(bundle.feature_names[name]),
        )
    return prepared


def instantiate_controller(
    feature_set: str,
    feature_dim: int,
    hidden: int,
    dropout: float,
    tmin: float,
    tmax: float,
    pmin: float,
    pmax: float,
) -> nn.Module:
    if feature_set == "mean":
        return MeanController(tmin, tmax, pmin, pmax)
    return MLPController(feature_dim, hidden, tmin, tmax, pmin, pmax, dropout=dropout)


def build_checkpoint_payload(
    model: nn.Module,
    *,
    feature_set: str,
    prepared: PreparedFeatureSet,
    source_path: str,
    hidden: int,
    dropout: float,
    profile_k: int,
    dist_k: int,
    val_mod: int,
    quant_step: float,
    tmin: float,
    tmax: float,
    pmin: float,
    pmax: float,
) -> Dict[str, object]:
    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {
        "format": "shadow_controller_checkpoint_v1",
        "state_dict": state_dict,
        "metadata": {
            "model_name": feature_set,
            "feature_set": feature_set,
            "model_type": "mean" if feature_set == "mean" else "mlp",
            "source_path": str(source_path),
            "feature_dim": 0 if prepared.X is None else int(prepared.X.shape[1]),
            "hidden": int(hidden),
            "dropout": float(dropout),
            "profile_k": int(profile_k),
            "dist_k": int(dist_k),
            "val_mod": int(val_mod),
            "quant_step": float(quant_step),
            "tmin": float(tmin),
            "tmax": float(tmax),
            "pmin": float(pmin),
            "pmax": float(pmax),
            "n_binary": int(prepared.n_binary),
            "mu": prepared.mu.tolist(),
            "sd": prepared.sd.tolist(),
            "feature_names": list(prepared.feature_names),
        },
    }


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden: int
    dropout: float
    soft_tau: float
    loss_kl: float
    loss_ent: float
    loss_nuc: float
    loss_tp: float
    device: str
    quant_step: float
    eval_batch_size: int


@dataclass
class EvalResult:
    name: str
    train_best_epoch: int
    val_loss: float
    T_mae: float
    p_mae: float
    T_r2: float
    p_r2: float
    kl: float
    js: float
    entropy_mae: float
    entropy_r2: float
    nucleus_mae: float
    top1_mae: float


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_torch(x: Optional[np.ndarray], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


def evaluate_controller(
    model: nn.Module,
    name: str,
    X: Optional[np.ndarray],
    logits: np.ndarray,
    yT: np.ndarray,
    yp: np.ndarray,
    indices: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> EvalResult:
    model.eval()

    T_preds: List[np.ndarray] = []
    p_preds: List[np.ndarray] = []
    T_true: List[np.ndarray] = []
    p_true: List[np.ndarray] = []

    all_kl: List[np.ndarray] = []
    all_js: List[np.ndarray] = []
    all_H_pred: List[np.ndarray] = []
    all_H_true: List[np.ndarray] = []
    all_N_pred: List[np.ndarray] = []
    all_N_true: List[np.ndarray] = []
    all_top1_pred: List[np.ndarray] = []
    all_top1_true: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx in batched_indices(indices, cfg.eval_batch_size, shuffle=False, rng=np.random.RandomState(0)):
            xb = None if X is None else torch.from_numpy(X[batch_idx]).to(device=device, dtype=torch.float32)
            lb = torch.from_numpy(logits[batch_idx]).to(device=device, dtype=torch.float32)
            yTb = torch.from_numpy(yT[batch_idx]).to(device=device, dtype=torch.float32)
            ypb = torch.from_numpy(yp[batch_idx]).to(device=device, dtype=torch.float32)

            if isinstance(model, MeanController):
                T_pred, p_pred = model(None, n=lb.shape[0])
            else:
                assert xb is not None
                T_pred, p_pred = model(xb)

            # Quantize at eval time if requested.
            if cfg.quant_step > 0:
                step = cfg.quant_step
                T_pred = torch.round(T_pred / step) * step
                p_pred = torch.round(p_pred / step) * step

            q_true, st_true = hard_top_p_from_sorted_logits_torch(lb, yTb, ypb)
            q_pred, st_pred = hard_top_p_from_sorted_logits_torch(lb, T_pred, p_pred)

            kl_per = (q_true.clamp_min(1e-12) * (q_true.clamp_min(1e-12).log() - q_pred.clamp_min(1e-12).log())).sum(dim=-1)
            m = 0.5 * (q_true + q_pred)
            js_per = 0.5 * (q_true.clamp_min(1e-12) * (q_true.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1) \
                   + 0.5 * (q_pred.clamp_min(1e-12) * (q_pred.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1)

            T_preds.append(T_pred.cpu().numpy())
            p_preds.append(p_pred.cpu().numpy())
            T_true.append(yTb.cpu().numpy())
            p_true.append(ypb.cpu().numpy())

            all_kl.append(kl_per.cpu().numpy())
            all_js.append(js_per.cpu().numpy())
            all_H_pred.append(st_pred["entropy"].cpu().numpy())
            all_H_true.append(st_true["entropy"].cpu().numpy())
            all_N_pred.append(st_pred["nucleus_size"].cpu().numpy())
            all_N_true.append(st_true["nucleus_size"].cpu().numpy())
            all_top1_pred.append(st_pred["top1"].cpu().numpy())
            all_top1_true.append(st_true["top1"].cpu().numpy())

    T_pred = np.concatenate(T_preds)
    p_pred = np.concatenate(p_preds)
    T_true_np = np.concatenate(T_true)
    p_true_np = np.concatenate(p_true)
    kl_np = np.concatenate(all_kl)
    js_np = np.concatenate(all_js)
    H_pred = np.concatenate(all_H_pred)
    H_true = np.concatenate(all_H_true)
    N_pred = np.concatenate(all_N_pred)
    N_true = np.concatenate(all_N_true)
    top1_pred = np.concatenate(all_top1_pred)
    top1_true = np.concatenate(all_top1_true)

    return EvalResult(
        name=name,
        train_best_epoch=-1,
        val_loss=float(np.mean(kl_np)),
        T_mae=float(np.mean(np.abs(T_pred - T_true_np))),
        p_mae=float(np.mean(np.abs(p_pred - p_true_np))),
        T_r2=r2_np(T_true_np, T_pred),
        p_r2=r2_np(p_true_np, p_pred),
        kl=float(np.mean(kl_np)),
        js=float(np.mean(js_np)),
        entropy_mae=float(np.mean(np.abs(H_pred - H_true))),
        entropy_r2=r2_np(H_true, H_pred),
        nucleus_mae=float(np.mean(np.abs(N_pred - N_true))),
        top1_mae=float(np.mean(np.abs(top1_pred - top1_true))),
    )


def train_controller(
    name: str,
    model: nn.Module,
    X: Optional[np.ndarray],
    logits: np.ndarray,
    yT: np.ndarray,
    yp: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[nn.Module, EvalResult]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.RandomState(0)

    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        train_kls = []
        train_ent = []
        train_nuc = []
        train_tp = []

        for batch_idx in batched_indices(tr_idx, cfg.batch_size, shuffle=True, rng=rng):
            xb = None if X is None else torch.from_numpy(X[batch_idx]).to(device=device, dtype=torch.float32)
            lb = torch.from_numpy(logits[batch_idx]).to(device=device, dtype=torch.float32)
            yTb = torch.from_numpy(yT[batch_idx]).to(device=device, dtype=torch.float32)
            ypb = torch.from_numpy(yp[batch_idx]).to(device=device, dtype=torch.float32)

            if isinstance(model, MeanController):
                T_pred, p_pred = model(None, n=lb.shape[0])
            else:
                assert xb is not None
                T_pred, p_pred = model(xb)

            q_true_hard, st_true = hard_top_p_from_sorted_logits_torch(lb, yTb, ypb)
            q_pred_soft, st_pred_soft = soft_top_p_from_sorted_logits_torch(lb, T_pred, p_pred, soft_tau=cfg.soft_tau)

            loss_kl = batch_forward_kl(q_true_hard, q_pred_soft)
            loss_ent = torch.mean(torch.abs(st_pred_soft["entropy"] - st_true["entropy"]))
            loss_nuc = torch.mean(torch.abs(st_pred_soft["nucleus_size"] - st_true["nucleus_size"]))
            loss_tp = F.mse_loss(T_pred, yTb) + F.mse_loss(p_pred, ypb)

            loss = cfg.loss_kl * loss_kl + cfg.loss_ent * loss_ent + cfg.loss_nuc * loss_nuc + cfg.loss_tp * loss_tp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_kls.append(float(loss_kl.detach().cpu()))
            train_ent.append(float(loss_ent.detach().cpu()))
            train_nuc.append(float(loss_nuc.detach().cpu()))
            train_tp.append(float(loss_tp.detach().cpu()))

        val_res = evaluate_controller(model, name, X, logits, yT, yp, va_idx, cfg, device)
        logger.info(
            "[%s] epoch %02d/%02d | train loss=%.5f kl=%.5f ent=%.5f nuc=%.5f tp=%.5f | "
            "val KL=%.5f JS=%.5f entMAE=%.5f nucMAE=%.5f T_R2=%.4f p_R2=%.4f",
            name, epoch + 1, cfg.epochs,
            np.mean(train_losses), np.mean(train_kls), np.mean(train_ent), np.mean(train_nuc), np.mean(train_tp),
            val_res.kl, val_res.js, val_res.entropy_mae, val_res.nucleus_mae, val_res.T_r2, val_res.p_r2,
        )

        if val_res.kl < best_val:
            best_val = val_res.kl
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_res = evaluate_controller(model, name, X, logits, yT, yp, va_idx, cfg, device)
    final_res.train_best_epoch = best_epoch
    logger.info(
        "[%s] BEST epoch=%d | val KL=%.5f JS=%.5f | entMAE=%.5f nucMAE=%.5f top1MAE=%.5f | "
        "T_MAE=%.5f p_MAE=%.5f | T_R2=%.4f p_R2=%.4f Hpost_R2=%.4f",
        name, best_epoch,
        final_res.kl, final_res.js, final_res.entropy_mae, final_res.nucleus_mae, final_res.top1_mae,
        final_res.T_mae, final_res.p_mae, final_res.T_r2, final_res.p_r2, final_res.entropy_r2,
    )
    return model, final_res


# -----------------------------------------------------------------------------
# Hypothesis logging
# -----------------------------------------------------------------------------


def _format_evidence_value(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        abs_value = abs(value)
        if abs_value >= 1000.0 or (0.0 < abs_value < 1e-4):
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def log_hypothesis(logger: logging.Logger, hid: str, statement: str, holds: bool, evidence: Dict[str, float | int | str]) -> None:
    payload = {
        "hypothesis_id": hid,
        "statement": statement,
        "supported": bool(holds),
        "evidence": evidence,
    }
    verdict = "SUPPORTED" if holds else "NOT SUPPORTED"
    evidence_text = ", ".join(
        f"{key}={_format_evidence_value(value)}"
        for key, value in evidence.items()
    ) if evidence else "(none)"
    logger.info("HYPOTHESIS %s [%s] %s", hid, verdict, statement)
    logger.info("HYPOTHESIS_EVIDENCE %s | %s", hid, evidence_text)
    logger.info("HYPOTHESIS_JSON %s", json.dumps(payload, sort_keys=True))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to HF dataset saved with load_from_disk.")
    ap.add_argument("--out-dir", type=str, default="shadow_controller_runs/default")
    ap.add_argument("--profile-k", type=int, default=64, help="How many shifted top-k logits to use as features.")
    ap.add_argument("--dist-k", type=int, default=200, help="How many top logits to use to approximate the controlled distribution.")
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--eval-batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--soft-tau", type=float, default=0.02)
    ap.add_argument("--loss-kl", type=float, default=1.0)
    ap.add_argument("--loss-ent", type=float, default=0.25)
    ap.add_argument("--loss-nuc", type=float, default=0.10)
    ap.add_argument("--loss-tp", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--quant-step", type=float, default=1.0 / 128.0)
    ap.add_argument("--max-train-tokens", type=int, default=0, help="Optional cap on number of train tokens (0 = all).")
    ap.add_argument("--max-val-tokens", type=int, default=0, help="Optional cap on number of val tokens (0 = all).")
    # Hypothesis thresholds.
    ap.add_argument("--h0-mean-mass-threshold", type=float, default=0.97)
    ap.add_argument("--h0-frac-mass-threshold", type=float, default=0.90)
    ap.add_argument("--h0-row-mass-threshold", type=float, default=0.95)
    ap.add_argument("--h4-js-threshold", type=float, default=0.05)
    ap.add_argument("--h4-entropy-mae-threshold", type=float, default=0.15)
    ap.add_argument("--h4-nucleus-mae-threshold", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logging(args.out_dir)
    set_seed(args.seed)
    device = pick_device(args.device)

    logger.info("Loading dataset from %s", args.path)
    bundle = build_feature_bundle(args.path, profile_k=args.profile_k, dist_k=args.dist_k, logger=logger)
    tr_mask, va_mask = split_by_seq_mod(bundle.seq_id, args.val_mod)
    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)

    if args.max_train_tokens > 0 and len(tr_idx) > args.max_train_tokens:
        rng = np.random.RandomState(args.seed)
        tr_idx = rng.choice(tr_idx, size=args.max_train_tokens, replace=False)
        tr_idx.sort()
    if args.max_val_tokens > 0 and len(va_idx) > args.max_val_tokens:
        rng = np.random.RandomState(args.seed + 1)
        va_idx = rng.choice(va_idx, size=args.max_val_tokens, replace=False)
        va_idx.sort()

    logger.info("Train rows: %d | Val rows: %d | dist_k=%d | profile_k=%d | device=%s",
                len(tr_idx), len(va_idx), args.dist_k, args.profile_k, device)

    # H0: Are top-k snapshots rich enough to approximate the full LM distribution?
    if bundle.mass_ref is not None:
        mean_mass = float(np.mean(bundle.mass_ref[va_idx]))
        frac_good = float(np.mean(bundle.mass_ref[va_idx] >= args.h0_row_mass_threshold))
        log_hypothesis(
            logger,
            hid="H0",
            statement="The stored top-k logits retain enough LM probability mass to approximate the controlled distribution.",
            holds=(mean_mass >= args.h0_mean_mass_threshold and frac_good >= args.h0_frac_mass_threshold),
            evidence={
                "dist_k": args.dist_k,
                "mean_retained_mass_at_T1": mean_mass,
                f"frac_rows_mass_ge_{args.h0_row_mass_threshold:.2f}": frac_good,
                "threshold_mean": args.h0_mean_mass_threshold,
                "threshold_frac": args.h0_frac_mass_threshold,
            },
        )

    prepared = prepare_feature_sets(bundle, tr_mask)

    tmin, tmax = float(bundle.yT[tr_idx].min()), float(bundle.yT[tr_idx].max())
    pmin, pmax = float(bundle.yp[tr_idx].min()), float(bundle.yp[tr_idx].max())
    logger.info("Observed train ranges | T_hat:[%.5f, %.5f] p_hat:[%.5f, %.5f]", tmin, tmax, pmin, pmax)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        dropout=args.dropout,
        soft_tau=args.soft_tau,
        loss_kl=args.loss_kl,
        loss_ent=args.loss_ent,
        loss_nuc=args.loss_nuc,
        loss_tp=args.loss_tp,
        device=str(device),
        quant_step=args.quant_step,
    )

    models: List[Tuple[str, Optional[np.ndarray], nn.Module]] = []
    for name in ("mean", "scalar", "scalar_bits", "scalar_bits_topk"):
        prepared_set = prepared[name]
        feature_dim = 0 if prepared_set.X is None else prepared_set.X.shape[1]
        models.append(
            (
                name,
                prepared_set.X,
                instantiate_controller(
                    feature_set=name,
                    feature_dim=feature_dim,
                    hidden=cfg.hidden,
                    dropout=cfg.dropout,
                    tmin=tmin,
                    tmax=tmax,
                    pmin=pmin,
                    pmax=pmax,
                ),
            )
        )

    all_results: Dict[str, EvalResult] = {}
    for name, X, model in models:
        logger.info("\n===== Training model: %s =====", name)
        trained_model, res = train_controller(
            name=name,
            model=model,
            X=X,
            logits=bundle.topk_logits,
            yT=bundle.yT,
            yp=bundle.yp,
            tr_idx=tr_idx,
            va_idx=va_idx,
            cfg=cfg,
            device=device,
            logger=logger,
        )
        all_results[name] = res
        checkpoint_path = os.path.join(args.out_dir, f"{name}_checkpoint.pt")
        checkpoint_payload = build_checkpoint_payload(
            trained_model,
            feature_set=name,
            prepared=prepared[name],
            source_path=args.path,
            hidden=args.hidden,
            dropout=args.dropout,
            profile_k=args.profile_k,
            dist_k=args.dist_k,
            val_mod=args.val_mod,
            quant_step=args.quant_step,
            tmin=tmin,
            tmax=tmax,
            pmin=pmin,
            pmax=pmax,
        )
        torch.save(checkpoint_payload, checkpoint_path)
        logger.info("Saved %s checkpoint to %s", name, checkpoint_path)

    # Hypothesis tests.
    mean_res = all_results["mean"]
    scalar_res = all_results["scalar"]
    sb_res = all_results["scalar_bits"]
    sbt_res = all_results["scalar_bits_topk"]

    log_hypothesis(
        logger,
        hid="H1",
        statement="Simple scalar uncertainty statistics contain more information about AutoDeco's behavior than a constant controller.",
        holds=(scalar_res.js < mean_res.js and scalar_res.entropy_mae < mean_res.entropy_mae),
        evidence={
            "mean_js": mean_res.js,
            "scalar_js": scalar_res.js,
            "mean_entropy_mae": mean_res.entropy_mae,
            "scalar_entropy_mae": scalar_res.entropy_mae,
            "mean_nucleus_mae": mean_res.nucleus_mae,
            "scalar_nucleus_mae": scalar_res.nucleus_mae,
        },
    )

    log_hypothesis(
        logger,
        hid="H2",
        statement="Position and formatting bits add incremental predictive signal beyond scalar uncertainty features.",
        holds=(sb_res.js < scalar_res.js and sb_res.entropy_mae <= scalar_res.entropy_mae),
        evidence={
            "scalar_js": scalar_res.js,
            "scalar_bits_js": sb_res.js,
            "scalar_entropy_mae": scalar_res.entropy_mae,
            "scalar_bits_entropy_mae": sb_res.entropy_mae,
            "scalar_nucleus_mae": scalar_res.nucleus_mae,
            "scalar_bits_nucleus_mae": sb_res.nucleus_mae,
        },
    )

    log_hypothesis(
        logger,
        hid="H3",
        statement="A low-dimensional top-k logit profile adds signal beyond summary statistics and structural bits.",
        holds=(sbt_res.js < sb_res.js and sbt_res.entropy_mae <= sb_res.entropy_mae),
        evidence={
            "scalar_bits_js": sb_res.js,
            "scalar_bits_topk_js": sbt_res.js,
            "scalar_bits_entropy_mae": sb_res.entropy_mae,
            "scalar_bits_topk_entropy_mae": sbt_res.entropy_mae,
            "scalar_bits_nucleus_mae": sb_res.nucleus_mae,
            "scalar_bits_topk_nucleus_mae": sbt_res.nucleus_mae,
        },
    )

    log_hypothesis(
        logger,
        hid="H4",
        statement="The best logits-only shadow controller closely matches AutoDeco's induced distribution on held-out tokens.",
        holds=(sbt_res.js <= args.h4_js_threshold and sbt_res.entropy_mae <= args.h4_entropy_mae_threshold and sbt_res.nucleus_mae <= args.h4_nucleus_mae_threshold),
        evidence={
            "best_js": sbt_res.js,
            "best_entropy_mae": sbt_res.entropy_mae,
            "best_nucleus_mae": sbt_res.nucleus_mae,
            "js_threshold": args.h4_js_threshold,
            "entropy_mae_threshold": args.h4_entropy_mae_threshold,
            "nucleus_mae_threshold": args.h4_nucleus_mae_threshold,
        },
    )

    log_hypothesis(
        logger,
        hid="H5",
        statement="Post-control behavior is more recoverable than raw control knobs, so raw T/p regression understates behavioral clonability.",
        holds=(sbt_res.entropy_r2 > max(sbt_res.T_r2, sbt_res.p_r2)),
        evidence={
            "best_T_r2": sbt_res.T_r2,
            "best_p_r2": sbt_res.p_r2,
            "best_post_entropy_r2": sbt_res.entropy_r2,
            "best_js": sbt_res.js,
        },
    )

    # Save metrics summary.
    summary = {
        name: {
            "best_epoch": res.train_best_epoch,
            "T_mae": res.T_mae,
            "p_mae": res.p_mae,
            "T_r2": res.T_r2,
            "p_r2": res.p_r2,
            "kl": res.kl,
            "js": res.js,
            "entropy_mae": res.entropy_mae,
            "entropy_r2": res.entropy_r2,
            "nucleus_mae": res.nucleus_mae,
            "top1_mae": res.top1_mae,
        }
        for name, res in all_results.items()
    }
    latex_rows = build_shadow_latex_rows(summary)
    summary["latex_rows"] = {
        "shadow_controller_table_rows": latex_rows,
    }

    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metrics summary to %s", summary_path)

    latex_rows_path = os.path.join(args.out_dir, "shadow_controller_table_rows.tex")
    with open(latex_rows_path, "w") as f:
        f.write(
            "% Ready-to-copy rows for colm2026_v5.tex.\n"
            "% Best value in each metric column is bolded automatically.\n"
            "% Columns: JS, Ent. MAE, Nuc. MAE, Top1 MAE, T-MAE, p-MAE, Post-Entropy R^2.\n"
        )
        f.write("\n".join(latex_rows))
        f.write("\n")
    logger.info("Saved LaTeX rows to %s", latex_rows_path)


if __name__ == "__main__":
    main()
