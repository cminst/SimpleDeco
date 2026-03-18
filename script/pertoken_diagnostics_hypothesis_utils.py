from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_from_disk


@dataclass
class DiagnosticsSlice:
    seq_id: np.ndarray
    token_id: np.ndarray
    T_hat: np.ndarray
    p_hat: np.ndarray
    H: np.ndarray
    H_norm: np.ndarray
    p_max: np.ndarray
    gap12: np.ndarray
    topk_ids: np.ndarray
    topk_logits: np.ndarray
    mass_ref: Optional[np.ndarray]
    gold_pos: np.ndarray


@dataclass
class TrainOperatingPoint:
    mean_T: float
    mean_p: float
    mean_beta: float
    T_from_mean_beta: float


@dataclass
class InverseTempTerms:
    covered_mask: np.ndarray
    covered_indices: np.ndarray
    beta_t: np.ndarray
    delta: np.ndarray
    z_gold: np.ndarray
    grad: np.ndarray
    var: np.ndarray
    alignment: np.ndarray
    penalty: np.ndarray
    predicted_net: np.ndarray
    actual_gain: np.ndarray


def as_2d_float32(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.float32)
    return np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)


def as_2d_int64(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.int64)
    return np.stack([np.asarray(x, dtype=np.int64) for x in arr], axis=0)


def split_by_seq_mod(seq_id: np.ndarray, val_mod: int) -> tuple[np.ndarray, np.ndarray]:
    val_mask = (seq_id % val_mod) == 0
    tr_mask = ~val_mask
    return tr_mask, val_mask


def rank_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if num_bins <= 1:
        return np.zeros(len(values), dtype=np.int32)
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int32)
    for bin_idx, idx in enumerate(np.array_split(order, num_bins)):
        bins[idx] = bin_idx
    return bins


def permute_within_groups(groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    perm = np.arange(len(groups), dtype=np.int64)
    for group_id in np.unique(groups):
        idx = np.flatnonzero(groups == group_id)
        if len(idx) <= 1:
            continue
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def select_mass_ref(data: Dict[str, np.ndarray], dist_k: int) -> Optional[np.ndarray]:
    if dist_k >= 200 and "mass200" in data:
        return data["mass200"].astype(np.float32)
    if dist_k >= 50 and "mass50" in data:
        return data["mass50"].astype(np.float32)
    if dist_k >= 10 and "mass10" in data:
        return data["mass10"].astype(np.float32)
    return None


def find_gold_positions(token_id: np.ndarray, topk_ids: np.ndarray) -> np.ndarray:
    matches = topk_ids == token_id[:, None]
    pos = matches.argmax(axis=1).astype(np.int32)
    pos[~matches.any(axis=1)] = -1
    return pos


def sort_topk_pairs_if_needed(
    topk_logits: np.ndarray,
    topk_ids: np.ndarray,
    logger: logging.Logger,
    sample_rows: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    n = topk_logits.shape[0]
    m = min(n, sample_rows)
    sample = topk_logits[:m]
    unsorted_frac = float(np.mean(np.any(sample[:, 1:] > sample[:, :-1], axis=1)))
    if unsorted_frac == 0.0:
        logger.info("Top-k logits appear pre-sorted descending.")
        return topk_logits, topk_ids

    order = np.argsort(-topk_logits, axis=1)
    topk_logits = np.take_along_axis(topk_logits, order, axis=1)
    topk_ids = np.take_along_axis(topk_ids, order, axis=1)
    logger.info("Detected unsorted top-k logits; sorted logits and ids descending once during preprocessing.")
    return topk_logits, topk_ids


def softmax_from_beta(sorted_logits: np.ndarray, beta: np.ndarray | float) -> np.ndarray:
    beta_arr = np.asarray(beta, dtype=np.float32)
    scaled = sorted_logits * beta_arr[:, None] if beta_arr.ndim == 1 else sorted_logits * beta_arr
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def rowwise_logsumexp(values: np.ndarray) -> np.ndarray:
    vmax = np.max(values, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(values - vmax), axis=1)) + vmax[:, 0]


def rowwise_nucleus_size_from_probs(sorted_probs: np.ndarray, top_p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(top_p, dtype=np.float32)
    if p.ndim == 0:
        p = np.full(sorted_probs.shape[0], float(p), dtype=np.float32)
    cdf = np.cumsum(sorted_probs, axis=1)
    return 1 + np.sum(cdf < p[:, None], axis=1)


def heldout_topk_adequacy_stats(
    bundle: DiagnosticsSlice,
    row_mass_threshold: float,
) -> Dict[str, float]:
    mean_mass = float(np.mean(bundle.mass_ref)) if bundle.mass_ref is not None else float("nan")
    frac_good = float(np.mean(bundle.mass_ref >= row_mass_threshold)) if bundle.mass_ref is not None else float("nan")
    gold_coverage = float(np.mean(bundle.gold_pos >= 0))
    return {
        "mean_retained_mass_at_T1": mean_mass,
        f"frac_rows_mass_ge_{row_mass_threshold:.2f}": frac_good,
        "gold_in_topk_rate": gold_coverage,
    }


def compute_inverse_temp_terms(bundle: DiagnosticsSlice, beta_bar: float) -> InverseTempTerms:
    covered_mask = bundle.gold_pos >= 0
    covered_indices = np.flatnonzero(covered_mask)
    if len(covered_indices) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return InverseTempTerms(
            covered_mask=covered_mask,
            covered_indices=covered_indices,
            beta_t=empty,
            delta=empty,
            z_gold=empty,
            grad=empty,
            var=empty,
            alignment=empty,
            penalty=empty,
            predicted_net=empty,
            actual_gain=empty,
        )
    logits = bundle.topk_logits[covered_mask]
    gold_pos = bundle.gold_pos[covered_mask]
    beta_t = 1.0 / np.clip(bundle.T_hat[covered_mask], 1e-6, None)
    delta = beta_t - beta_bar

    probs_bar = softmax_from_beta(logits, np.full(len(logits), beta_bar, dtype=np.float32))
    mean_z = np.sum(probs_bar * logits, axis=1)
    mean_z2 = np.sum(probs_bar * np.square(logits), axis=1)
    var = np.maximum(mean_z2 - np.square(mean_z), 0.0)
    z_gold = logits[np.arange(len(logits)), gold_pos]
    grad = z_gold - mean_z
    alignment = grad * delta
    penalty = 0.5 * var * np.square(delta)
    predicted_net = alignment - penalty

    logsumexp_auto = rowwise_logsumexp(logits * beta_t[:, None])
    logsumexp_bar = rowwise_logsumexp(logits * beta_bar)
    actual_gain = beta_t * z_gold - logsumexp_auto - (beta_bar * z_gold - logsumexp_bar)

    return InverseTempTerms(
        covered_mask=covered_mask,
        covered_indices=covered_indices,
        beta_t=beta_t.astype(np.float32),
        delta=delta.astype(np.float32),
        z_gold=z_gold.astype(np.float32),
        grad=grad.astype(np.float32),
        var=var.astype(np.float32),
        alignment=alignment.astype(np.float32),
        penalty=penalty.astype(np.float32),
        predicted_net=predicted_net.astype(np.float32),
        actual_gain=actual_gain.astype(np.float32),
    )


def load_diagnostics_slice(
    ds_path: str,
    dist_k: int,
    val_mod: int,
    seed: int,
    logger: logging.Logger,
    max_val_tokens: int = 0,
    use_all_tokens: bool = False,
) -> tuple[DiagnosticsSlice, TrainOperatingPoint]:
    ds = load_from_disk(ds_path)
    if "tokens" not in ds:
        raise ValueError("Expected a DatasetDict with a 'tokens' split.")
    tok = ds["tokens"]

    meta_cols = [
        "seq_id", "token_id", "T_hat", "p_hat", "H", "H_norm", "p_max", "gap12",
        "mass10", "mass50", "mass200",
    ]
    needed_cols = meta_cols + ["topk_ids", "topk_logits"]
    missing = [c for c in needed_cols if c not in tok.features]
    if missing:
        raise ValueError(
            "Dataset is missing required columns for these diagnostics: "
            f"{missing}. Rerun collect_pertoken_diagnostics.py with token_id and top-k sketches enabled."
        )

    meta = tok.select_columns(meta_cols).with_format("numpy")[:]
    seq_id = meta["seq_id"].astype(np.int64)
    if use_all_tokens:
        tr_idx = np.arange(len(seq_id), dtype=np.int64)
        va_idx = tr_idx.copy()
        logger.info(
            "Using all %d diagnostics rows for both operating-point estimation and evaluation.",
            len(seq_id),
        )
    else:
        tr_mask, va_mask = split_by_seq_mod(seq_id, val_mod)
        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"val_mod={val_mod} produced an empty train or validation split.")

    T_train = meta["T_hat"][tr_idx].astype(np.float32)
    p_train = meta["p_hat"][tr_idx].astype(np.float32)
    beta_train = 1.0 / np.clip(T_train, 1e-6, None)
    op = TrainOperatingPoint(
        mean_T=float(np.mean(T_train)),
        mean_p=float(np.mean(p_train)),
        mean_beta=float(np.mean(beta_train)),
        T_from_mean_beta=float(1.0 / np.mean(beta_train)),
    )

    if max_val_tokens > 0 and len(va_idx) > max_val_tokens:
        rng = np.random.RandomState(seed)
        va_idx = rng.choice(va_idx, size=max_val_tokens, replace=False)
        va_idx.sort()

    val_ds = tok.select(va_idx.tolist()).select_columns(
        ["seq_id", "token_id", "T_hat", "p_hat", "H", "H_norm", "p_max", "gap12", "mass10", "mass50", "mass200", "topk_ids", "topk_logits"]
    ).with_format("numpy")
    val = val_ds[:]
    topk_logits = as_2d_float32(val["topk_logits"])
    topk_ids = as_2d_int64(val["topk_ids"])
    if topk_logits.shape != topk_ids.shape:
        raise ValueError(
            f"topk_logits shape {topk_logits.shape} does not match topk_ids shape {topk_ids.shape}."
        )
    if dist_k > topk_logits.shape[1]:
        raise ValueError(
            f"Requested dist_k={dist_k}, but the dataset only stores {topk_logits.shape[1]} top-k entries."
        )

    topk_logits = topk_logits[:, :dist_k]
    topk_ids = topk_ids[:, :dist_k]
    topk_logits, topk_ids = sort_topk_pairs_if_needed(topk_logits, topk_ids, logger=logger)
    gold_pos = find_gold_positions(val["token_id"].astype(np.int64), topk_ids)

    bundle = DiagnosticsSlice(
        seq_id=val["seq_id"].astype(np.int64),
        token_id=val["token_id"].astype(np.int64),
        T_hat=val["T_hat"].astype(np.float32),
        p_hat=val["p_hat"].astype(np.float32),
        H=val["H"].astype(np.float32),
        H_norm=val["H_norm"].astype(np.float32),
        p_max=val["p_max"].astype(np.float32),
        gap12=val["gap12"].astype(np.float32),
        topk_ids=topk_ids,
        topk_logits=topk_logits,
        mass_ref=select_mass_ref(val, dist_k=dist_k),
        gold_pos=gold_pos,
    )
    return bundle, op
