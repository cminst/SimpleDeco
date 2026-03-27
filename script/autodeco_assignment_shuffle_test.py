#!/usr/bin/env python3
"""
Teacher-forced shuffle test for AutoDeco's per-token control assignment.

The core comparison is:
  1. AutoDeco:      use the stored per-token (T_hat, p_hat)
  2. MeanShift:     use the train-split mean operating point (T_bar, p_bar)
  3. Shuffle:       permute the same (T_hat, p_hat) pairs across held-out tokens
  4. BinShuffle:    permute within coarse H_norm / p_max bins

This script is intentionally shaped like shadow_controller_dist_match.py:
- it loads the saved per-token diagnostics dataset from disk
- it evaluates pre-registered hypotheses on a held-out sequence split
- it logs human-readable hypothesis verdicts and saves JSON summaries

Usage example
-------------
python3 script/autodeco_assignment_shuffle_test.py \
    --path ckpt/pertoken_diagnostics/autodeco_qwen7b_dolci_val_balanced/ \
    --dist-k 200 \
    --shuffle-repeats 8 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_from_disk

from shadow_controller_dist_match import (
    hard_top_p_from_sorted_logits_torch,
    log_hypothesis,
    pick_device,
    set_seed,
    setup_logging,
    safe_div,
)


@dataclass
class TokenSlice:
    seq_id: np.ndarray
    token_id: np.ndarray
    T_hat: np.ndarray
    p_hat: np.ndarray
    H_norm: np.ndarray
    p_max: np.ndarray
    topk_ids: np.ndarray
    topk_logits: np.ndarray
    mass_ref: Optional[np.ndarray]
    gold_pos: np.ndarray


@dataclass
class PolicyMetrics:
    name: str
    mean_T: float
    std_T: float
    mean_p: float
    std_p: float
    mean_entropy: float
    mean_nucleus_size: float
    gold_covered_rate: float
    gold_in_nucleus_rate: Optional[float]
    gold_logprob_mean: Optional[float]
    gold_logprob_median: Optional[float]
    js_to_auto: float
    kl_auto_to_policy: float
    entropy_absdiff: float
    nucleus_absdiff: float
    top1_absdiff: float
    support_change_rate: float
    nucleus_delta_gt1_rate: float
    nucleus_delta_gt5_rate: float
    argmax_change_rate: float


def _as_2d_float32(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.float32)
    return np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)


def _as_2d_int64(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
        return arr.astype(np.int64)
    return np.stack([np.asarray(x, dtype=np.int64) for x in arr], axis=0)


def _sort_topk_pairs_if_needed(
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


def split_by_seq_mod(seq_id: np.ndarray, val_mod: int) -> tuple[np.ndarray, np.ndarray]:
    val_mask = (seq_id % val_mod) == 0
    tr_mask = ~val_mask
    return tr_mask, val_mask


def _rank_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    if num_bins <= 1:
        return np.zeros(len(values), dtype=np.int32)
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int32)
    for bin_idx, idx in enumerate(np.array_split(order, num_bins)):
        bins[idx] = bin_idx
    return bins


def _permute_within_groups(groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    perm = np.arange(len(groups), dtype=np.int64)
    for group_id in np.unique(groups):
        idx = np.flatnonzero(groups == group_id)
        if len(idx) <= 1:
            continue
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def _select_mass_ref(data: Dict[str, np.ndarray], dist_k: int) -> Optional[np.ndarray]:
    if dist_k >= 200 and "mass200" in data:
        return data["mass200"].astype(np.float32)
    if dist_k >= 50 and "mass50" in data:
        return data["mass50"].astype(np.float32)
    if dist_k >= 10 and "mass10" in data:
        return data["mass10"].astype(np.float32)
    return None


def _find_gold_positions(token_id: np.ndarray, topk_ids: np.ndarray) -> np.ndarray:
    matches = topk_ids == token_id[:, None]
    pos = matches.argmax(axis=1).astype(np.int32)
    pos[~matches.any(axis=1)] = -1
    return pos


def load_eval_slice(
    ds_path: str,
    dist_k: int,
    val_mod: int,
    seed: int,
    logger: logging.Logger,
    max_val_tokens: int = 0,
) -> tuple[TokenSlice, float, float]:
    ds = load_from_disk(ds_path)
    if "tokens" not in ds:
        raise ValueError("Expected a DatasetDict with a 'tokens' split.")
    tok = ds["tokens"]

    meta_cols = [
        "seq_id", "token_id", "T_hat", "p_hat", "H_norm", "p_max", "mass10", "mass50", "mass200",
    ]
    needed_cols = meta_cols + ["topk_ids", "topk_logits"]
    missing = [c for c in needed_cols if c not in tok.features]
    if missing:
        raise ValueError(
            "Dataset is missing required columns for the shuffle test: "
            f"{missing}. Rerun collect_pertoken_diagnostics.py with token_id and top-k sketches enabled."
        )

    meta = tok.select_columns(meta_cols).with_format("numpy")[:]
    seq_id = meta["seq_id"].astype(np.int64)
    tr_mask, va_mask = split_by_seq_mod(seq_id, val_mod)
    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise ValueError(f"val_mod={val_mod} produced an empty train or validation split.")

    mean_T = float(np.mean(meta["T_hat"][tr_idx].astype(np.float32)))
    mean_p = float(np.mean(meta["p_hat"][tr_idx].astype(np.float32)))

    if max_val_tokens > 0 and len(va_idx) > max_val_tokens:
        rng = np.random.RandomState(seed)
        va_idx = rng.choice(va_idx, size=max_val_tokens, replace=False)
        va_idx.sort()

    val_ds = tok.select(va_idx.tolist()).select_columns(
        ["seq_id", "token_id", "T_hat", "p_hat", "H_norm", "p_max", "mass10", "mass50", "mass200", "topk_ids", "topk_logits"]
    ).with_format("numpy")
    val = val_ds[:]
    topk_logits = _as_2d_float32(val["topk_logits"])
    topk_ids = _as_2d_int64(val["topk_ids"])
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
    topk_logits, topk_ids = _sort_topk_pairs_if_needed(topk_logits, topk_ids, logger=logger)
    gold_pos = _find_gold_positions(val["token_id"].astype(np.int64), topk_ids)

    bundle = TokenSlice(
        seq_id=val["seq_id"].astype(np.int64),
        token_id=val["token_id"].astype(np.int64),
        T_hat=val["T_hat"].astype(np.float32),
        p_hat=val["p_hat"].astype(np.float32),
        H_norm=val["H_norm"].astype(np.float32),
        p_max=val["p_max"].astype(np.float32),
        topk_ids=topk_ids,
        topk_logits=topk_logits,
        mass_ref=_select_mass_ref(val, dist_k=dist_k),
        gold_pos=gold_pos,
    )
    return bundle, mean_T, mean_p


def evaluate_policy(
    name: str,
    bundle: TokenSlice,
    T_policy: np.ndarray,
    p_policy: np.ndarray,
    batch_size: int,
    device: torch.device,
    compare_to_auto: bool = True,
) -> PolicyMetrics:
    n = len(bundle.T_hat)
    sum_entropy = 0.0
    sum_nucleus = 0.0
    sum_js = 0.0
    sum_kl = 0.0
    sum_abs_ent = 0.0
    sum_abs_nucleus = 0.0
    sum_abs_top1 = 0.0
    count_support_change = 0
    count_nucleus_gt1 = 0
    count_nucleus_gt5 = 0
    gold_covered = 0
    gold_in_nucleus = 0
    gold_logprobs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            lb = torch.from_numpy(bundle.topk_logits[start:end]).to(device=device, dtype=torch.float32)
            Tb = torch.from_numpy(T_policy[start:end]).to(device=device, dtype=torch.float32)
            pb = torch.from_numpy(p_policy[start:end]).to(device=device, dtype=torch.float32)

            q_policy, st_policy = hard_top_p_from_sorted_logits_torch(lb, Tb, pb)
            sum_entropy += float(st_policy["entropy"].sum().cpu())
            sum_nucleus += float(st_policy["nucleus_size"].sum().cpu())

            if compare_to_auto:
                Ta = torch.from_numpy(bundle.T_hat[start:end]).to(device=device, dtype=torch.float32)
                pa = torch.from_numpy(bundle.p_hat[start:end]).to(device=device, dtype=torch.float32)
                q_auto, st_auto = hard_top_p_from_sorted_logits_torch(lb, Ta, pa)

                q_auto_safe = q_auto.clamp_min(1e-12)
                q_policy_safe = q_policy.clamp_min(1e-12)
                m = 0.5 * (q_auto_safe + q_policy_safe)
                js_per = 0.5 * (q_auto_safe * (q_auto_safe.log() - m.log())).sum(dim=-1) \
                    + 0.5 * (q_policy_safe * (q_policy_safe.log() - m.log())).sum(dim=-1)
                kl_per = (q_auto_safe * (q_auto_safe.log() - q_policy_safe.log())).sum(dim=-1)
                nucleus_delta = torch.abs(st_policy["nucleus_size"] - st_auto["nucleus_size"])

                sum_js += float(js_per.sum().cpu())
                sum_kl += float(kl_per.sum().cpu())
                sum_abs_ent += float(torch.abs(st_policy["entropy"] - st_auto["entropy"]).sum().cpu())
                sum_abs_nucleus += float(nucleus_delta.sum().cpu())
                sum_abs_top1 += float(torch.abs(st_policy["top1"] - st_auto["top1"]).sum().cpu())
                count_support_change += int((nucleus_delta > 0).sum().cpu())
                count_nucleus_gt1 += int((nucleus_delta > 1).sum().cpu())
                count_nucleus_gt5 += int((nucleus_delta > 5).sum().cpu())

            gold_pos = bundle.gold_pos[start:end]
            covered_idx = np.flatnonzero(gold_pos >= 0)
            if len(covered_idx) > 0:
                row_t = torch.from_numpy(covered_idx).to(device=device, dtype=torch.long)
                pos_t = torch.from_numpy(gold_pos[covered_idx]).to(device=device, dtype=torch.long)
                gold_q = q_policy.index_select(0, row_t)[torch.arange(len(covered_idx), device=device), pos_t]
                gold_logprobs.append(torch.log(gold_q.clamp_min(1e-12)).cpu().numpy())
                gold_in_nucleus += int((gold_q > 0).sum().cpu())
                gold_covered += len(covered_idx)

    gold_logprob_np = np.concatenate(gold_logprobs) if gold_logprobs else np.empty((0,), dtype=np.float32)
    return PolicyMetrics(
        name=name,
        mean_T=float(np.mean(T_policy)),
        std_T=float(np.std(T_policy)),
        mean_p=float(np.mean(p_policy)),
        std_p=float(np.std(p_policy)),
        mean_entropy=safe_div(sum_entropy, n),
        mean_nucleus_size=safe_div(sum_nucleus, n),
        gold_covered_rate=safe_div(gold_covered, n),
        gold_in_nucleus_rate=(safe_div(gold_in_nucleus, gold_covered) if gold_covered > 0 else None),
        gold_logprob_mean=(float(np.mean(gold_logprob_np)) if gold_logprob_np.size > 0 else None),
        gold_logprob_median=(float(np.median(gold_logprob_np)) if gold_logprob_np.size > 0 else None),
        js_to_auto=(safe_div(sum_js, n) if compare_to_auto else 0.0),
        kl_auto_to_policy=(safe_div(sum_kl, n) if compare_to_auto else 0.0),
        entropy_absdiff=(safe_div(sum_abs_ent, n) if compare_to_auto else 0.0),
        nucleus_absdiff=(safe_div(sum_abs_nucleus, n) if compare_to_auto else 0.0),
        top1_absdiff=(safe_div(sum_abs_top1, n) if compare_to_auto else 0.0),
        support_change_rate=(safe_div(count_support_change, n) if compare_to_auto else 0.0),
        nucleus_delta_gt1_rate=(safe_div(count_nucleus_gt1, n) if compare_to_auto else 0.0),
        nucleus_delta_gt5_rate=(safe_div(count_nucleus_gt5, n) if compare_to_auto else 0.0),
        argmax_change_rate=0.0,
    )


def compute_effect_size_summary(
    bundle: TokenSlice,
    mean_T: float,
    mean_p: float,
    batch_size: int,
    device: torch.device,
    uncertainty_bins: int,
) -> Dict[str, Any]:
    n = len(bundle.T_hat)
    js_parts: List[np.ndarray] = []
    abs_ent_parts: List[np.ndarray] = []
    abs_top1_parts: List[np.ndarray] = []
    abs_nucleus_parts: List[np.ndarray] = []

    T_mean = np.full(n, mean_T, dtype=np.float32)
    p_mean = np.full(n, mean_p, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            lb = torch.from_numpy(bundle.topk_logits[start:end]).to(device=device, dtype=torch.float32)
            Ta = torch.from_numpy(bundle.T_hat[start:end]).to(device=device, dtype=torch.float32)
            pa = torch.from_numpy(bundle.p_hat[start:end]).to(device=device, dtype=torch.float32)
            Tm = torch.from_numpy(T_mean[start:end]).to(device=device, dtype=torch.float32)
            pm = torch.from_numpy(p_mean[start:end]).to(device=device, dtype=torch.float32)

            q_auto, st_auto = hard_top_p_from_sorted_logits_torch(lb, Ta, pa)
            q_mean, st_mean = hard_top_p_from_sorted_logits_torch(lb, Tm, pm)
            q_auto_safe = q_auto.clamp_min(1e-12)
            q_mean_safe = q_mean.clamp_min(1e-12)
            m = 0.5 * (q_auto_safe + q_mean_safe)
            js_per = 0.5 * (q_auto_safe * (q_auto_safe.log() - m.log())).sum(dim=-1) \
                + 0.5 * (q_mean_safe * (q_mean_safe.log() - m.log())).sum(dim=-1)

            js_parts.append(js_per.cpu().numpy())
            abs_ent_parts.append(torch.abs(st_auto["entropy"] - st_mean["entropy"]).cpu().numpy())
            abs_top1_parts.append(torch.abs(st_auto["top1"] - st_mean["top1"]).cpu().numpy())
            abs_nucleus_parts.append(torch.abs(st_auto["nucleus_size"] - st_mean["nucleus_size"]).cpu().numpy())

    js = np.concatenate(js_parts)
    abs_ent = np.concatenate(abs_ent_parts)
    abs_top1 = np.concatenate(abs_top1_parts)
    abs_nucleus = np.concatenate(abs_nucleus_parts)
    bins = _rank_bins(bundle.H_norm, uncertainty_bins)

    by_bin = []
    for bin_idx in range(int(bins.max()) + 1):
        mask = bins == bin_idx
        if not np.any(mask):
            continue
        h_vals = bundle.H_norm[mask]
        bin_entry = {
            "bin_index": int(bin_idx),
            "count": int(mask.sum()),
            "h_norm_min": float(np.min(h_vals)),
            "h_norm_max": float(np.max(h_vals)),
            "js_mean": float(np.mean(js[mask])),
            "js_median": float(np.median(js[mask])),
            "entropy_absdiff_mean": float(np.mean(abs_ent[mask])),
            "top1_absdiff_mean": float(np.mean(abs_top1[mask])),
            "support_change_rate": float(np.mean(abs_nucleus[mask] > 0)),
            "nucleus_delta_gt1_rate": float(np.mean(abs_nucleus[mask] > 1)),
            "nucleus_delta_gt5_rate": float(np.mean(abs_nucleus[mask] > 5)),
        }
        by_bin.append(bin_entry)

    return {
        "token_count": int(n),
        "js_mean": float(np.mean(js)),
        "js_median": float(np.median(js)),
        "js_p90": float(np.quantile(js, 0.90)),
        "js_p99": float(np.quantile(js, 0.99)),
        "entropy_absdiff_mean": float(np.mean(abs_ent)),
        "entropy_absdiff_median": float(np.median(abs_ent)),
        "top1_absdiff_mean": float(np.mean(abs_top1)),
        "top1_absdiff_median": float(np.median(abs_top1)),
        "support_change_rate": float(np.mean(abs_nucleus > 0)),
        "nucleus_delta_gt1_rate": float(np.mean(abs_nucleus > 1)),
        "nucleus_delta_gt5_rate": float(np.mean(abs_nucleus > 5)),
        "argmax_change_rate": 0.0,
        "by_h_norm_bin": by_bin,
    }


def aggregate_policy_metrics(metrics_list: Sequence[PolicyMetrics]) -> Dict[str, Any]:
    if not metrics_list:
        raise ValueError("metrics_list must be non-empty")
    summary: Dict[str, Any] = {"name": metrics_list[0].name, "repeats": len(metrics_list)}
    for field_info in fields(PolicyMetrics):
        key = field_info.name
        if key == "name":
            continue
        values = [getattr(m, key) for m in metrics_list if getattr(m, key) is not None]
        if not values:
            summary[key] = None
            summary[f"{key}_std"] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=0))
    return summary


def _fmt_mean_std(mean: Optional[float], std: Optional[float], digits: int = 5) -> str:
    if mean is None:
        return "n/a"
    if std is None or std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def log_policy_summary(logger: logging.Logger, summary: Dict[str, Any]) -> None:
    logger.info(
        "POLICY %s | T=%s | p=%s | gold_logprob=%s | gold_in_nucleus=%s | "
        "entropy=%s | nucleus=%s | JS_to_auto=%s | support_change=%s",
        summary["name"],
        _fmt_mean_std(summary.get("mean_T"), summary.get("mean_T_std")),
        _fmt_mean_std(summary.get("mean_p"), summary.get("mean_p_std")),
        _fmt_mean_std(summary.get("gold_logprob_mean"), summary.get("gold_logprob_mean_std")),
        _fmt_mean_std(summary.get("gold_in_nucleus_rate"), summary.get("gold_in_nucleus_rate_std")),
        _fmt_mean_std(summary.get("mean_entropy"), summary.get("mean_entropy_std")),
        _fmt_mean_std(summary.get("mean_nucleus_size"), summary.get("mean_nucleus_size_std")),
        _fmt_mean_std(summary.get("js_to_auto"), summary.get("js_to_auto_std")),
        _fmt_mean_std(summary.get("support_change_rate"), summary.get("support_change_rate_std")),
    )


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to the saved per-token diagnostics DatasetDict.")
    ap.add_argument("--output-dir", dest="out_dir", type=str, default="shuffle_test_runs/default")
    ap.add_argument("--dist-k", type=int, default=200, help="How many stored top-k logits to use for the truncated distribution.")
    ap.add_argument("--val-mod", type=int, default=10, help="Validation split is seq_id %% val_mod == 0.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--shuffle-repeats", type=int, default=8)
    ap.add_argument("--shuffle-bins", type=int, default=10, help="Number of quantile bins for H_norm / p_max shuffles.")
    ap.add_argument("--effect-bins", type=int, default=3, help="Number of H_norm bins for the AutoDeco-vs-MeanShift effect summary.")
    ap.add_argument("--max-val-tokens", type=int, default=0, help="Optional cap on held-out tokens (0 = all).")
    ap.add_argument("--h0-mean-mass-threshold", type=float, default=0.97)
    ap.add_argument("--h0-frac-mass-threshold", type=float, default=0.90)
    ap.add_argument("--h0-row-mass-threshold", type=float, default=0.95)
    ap.add_argument("--h0-gold-coverage-threshold", type=float, default=0.80)
    ap.add_argument("--h1-js-ratio-threshold", type=float, default=0.75)
    ap.add_argument("--h1-gold-preserve-threshold", type=float, default=0.67)
    ap.add_argument("--h2-js-improvement-threshold", type=float, default=0.95)
    ap.add_argument("--h2-gold-tolerance", type=float, default=1e-4)
    ap.add_argument("--h3-median-js-threshold", type=float, default=0.01)
    ap.add_argument("--h3-support-change-threshold", type=float, default=0.25)
    ap.add_argument("--h3-large-nucleus-threshold", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logging(args.out_dir)
    set_seed(args.seed)
    device = pick_device(args.device)

    logger.info("Loading diagnostics dataset from %s", args.path)
    bundle, mean_T, mean_p = load_eval_slice(
        ds_path=args.path,
        dist_k=args.dist_k,
        val_mod=args.val_mod,
        seed=args.seed,
        logger=logger,
        max_val_tokens=args.max_val_tokens,
    )
    logger.info(
        "Held-out rows: %d | dist_k=%d | mean operating point: T=%.5f p=%.5f | device=%s",
        len(bundle.T_hat), args.dist_k, mean_T, mean_p, device,
    )

    if bundle.mass_ref is not None:
        mean_mass = float(np.mean(bundle.mass_ref))
        frac_good = float(np.mean(bundle.mass_ref >= args.h0_row_mass_threshold))
    else:
        mean_mass = float("nan")
        frac_good = float("nan")
    gold_coverage = float(np.mean(bundle.gold_pos >= 0))
    log_hypothesis(
        logger,
        hid="H0",
        statement="The stored top-k sketches are rich enough for the shuffle test to approximate the controlled distribution and gold-token metrics.",
        holds=(
            (bundle.mass_ref is None or (mean_mass >= args.h0_mean_mass_threshold and frac_good >= args.h0_frac_mass_threshold))
            and gold_coverage >= args.h0_gold_coverage_threshold
        ),
        evidence={
            "dist_k": args.dist_k,
            "mean_retained_mass_at_T1": mean_mass,
            f"frac_rows_mass_ge_{args.h0_row_mass_threshold:.2f}": frac_good,
            "gold_in_topk_rate": gold_coverage,
            "mass_mean_threshold": args.h0_mean_mass_threshold,
            "mass_frac_threshold": args.h0_frac_mass_threshold,
            "gold_coverage_threshold": args.h0_gold_coverage_threshold,
        },
    )
    if gold_coverage < args.h0_gold_coverage_threshold:
        logger.warning(
            "Gold-token coverage in the stored top-k sketch is only %.4f. If you need a stronger gold-logprob test, rerun collect_pertoken_diagnostics.py with a larger --topk_sketch.",
            gold_coverage,
        )

    mean_policy_T = np.full(len(bundle.T_hat), mean_T, dtype=np.float32)
    mean_policy_p = np.full(len(bundle.T_hat), mean_p, dtype=np.float32)

    logger.info("Evaluating deterministic baselines.")
    auto_metrics = evaluate_policy(
        name="autodeco",
        bundle=bundle,
        T_policy=bundle.T_hat,
        p_policy=bundle.p_hat,
        batch_size=args.batch_size,
        device=device,
        compare_to_auto=False,
    )
    mean_metrics = evaluate_policy(
        name="meanshift",
        bundle=bundle,
        T_policy=mean_policy_T,
        p_policy=mean_policy_p,
        batch_size=args.batch_size,
        device=device,
        compare_to_auto=True,
    )

    repeat_metrics: Dict[str, List[PolicyMetrics]] = {
        "shuffle_full": [],
        "shuffle_hnorm_bin": [],
        "shuffle_pmax_bin": [],
    }
    hnorm_bins = _rank_bins(bundle.H_norm, args.shuffle_bins)
    pmax_bins = _rank_bins(bundle.p_max, args.shuffle_bins)

    logger.info("Evaluating shuffled policies (%d repeats).", args.shuffle_repeats)
    for repeat_idx in range(args.shuffle_repeats):
        rng = np.random.RandomState(args.seed + 1000 + repeat_idx)
        perms = {
            "shuffle_full": rng.permutation(len(bundle.T_hat)),
            "shuffle_hnorm_bin": _permute_within_groups(hnorm_bins, rng),
            "shuffle_pmax_bin": _permute_within_groups(pmax_bins, rng),
        }
        for policy_name, perm in perms.items():
            repeat_metrics[policy_name].append(
                evaluate_policy(
                    name=policy_name,
                    bundle=bundle,
                    T_policy=bundle.T_hat[perm],
                    p_policy=bundle.p_hat[perm],
                    batch_size=args.batch_size,
                    device=device,
                    compare_to_auto=True,
                )
            )

    auto_summary = aggregate_policy_metrics([auto_metrics])
    mean_summary = aggregate_policy_metrics([mean_metrics])
    shuffle_summaries = {name: aggregate_policy_metrics(items) for name, items in repeat_metrics.items()}

    logger.info("\n=== Policy Summaries ===")
    for summary in [auto_summary, mean_summary, shuffle_summaries["shuffle_full"], shuffle_summaries["shuffle_hnorm_bin"], shuffle_summaries["shuffle_pmax_bin"]]:
        log_policy_summary(logger, summary)

    effect_summary = compute_effect_size_summary(
        bundle=bundle,
        mean_T=mean_T,
        mean_p=mean_p,
        batch_size=args.batch_size,
        device=device,
        uncertainty_bins=args.effect_bins,
    )
    logger.info(
        "AUTO_vs_MEAN | JS median=%.5f p90=%.5f p99=%.5f | |dH| mean=%.5f | |dtop1| mean=%.5f | "
        "support_change=%.5f | |dN|>1=%.5f | |dN|>5=%.5f | argmax_change=%.5f",
        effect_summary["js_median"],
        effect_summary["js_p90"],
        effect_summary["js_p99"],
        effect_summary["entropy_absdiff_mean"],
        effect_summary["top1_absdiff_mean"],
        effect_summary["support_change_rate"],
        effect_summary["nucleus_delta_gt1_rate"],
        effect_summary["nucleus_delta_gt5_rate"],
        effect_summary["argmax_change_rate"],
    )
    for bin_entry in effect_summary["by_h_norm_bin"]:
        logger.info(
            "AUTO_vs_MEAN_BIN %d | H_norm=[%.4f, %.4f] | count=%d | JS mean=%.5f median=%.5f | "
            "support_change=%.5f | |dN|>1=%.5f | |dN|>5=%.5f",
            bin_entry["bin_index"],
            bin_entry["h_norm_min"],
            bin_entry["h_norm_max"],
            bin_entry["count"],
            bin_entry["js_mean"],
            bin_entry["js_median"],
            bin_entry["support_change_rate"],
            bin_entry["nucleus_delta_gt1_rate"],
            bin_entry["nucleus_delta_gt5_rate"],
        )

    auto_gold = auto_summary.get("gold_logprob_mean")
    mean_gold = mean_summary.get("gold_logprob_mean")
    full_shuffle = shuffle_summaries["shuffle_full"]
    full_gold = full_shuffle.get("gold_logprob_mean")
    mean_js = mean_summary["js_to_auto"]
    full_js = full_shuffle["js_to_auto"]
    gold_gain = None
    gold_preserved = None
    gold_support_ok = True
    if auto_gold is not None and mean_gold is not None and full_gold is not None:
        gold_gain = float(auto_gold - mean_gold)
        if gold_gain > 1e-8:
            gold_preserved = float((full_gold - mean_gold) / gold_gain)
            gold_support_ok = gold_preserved >= args.h1_gold_preserve_threshold

    js_ratio = float(full_js / max(mean_js, 1e-12))
    log_hypothesis(
        logger,
        hid="H1",
        statement="Randomly reassigning AutoDeco's control pairs preserves most of the gain over MeanShift, so token-specific assignment has limited extra leverage.",
        holds=(js_ratio <= args.h1_js_ratio_threshold and gold_support_ok),
        evidence={
            "meanshift_js_to_auto": mean_js,
            "full_shuffle_js_to_auto": full_js,
            "full_shuffle_js_over_meanshift_js": js_ratio,
            "autodeco_gold_logprob_covered": (auto_gold if auto_gold is not None else "n/a"),
            "meanshift_gold_logprob_covered": (mean_gold if mean_gold is not None else "n/a"),
            "full_shuffle_gold_logprob_covered": (
                _fmt_mean_std(full_shuffle.get("gold_logprob_mean"), full_shuffle.get("gold_logprob_mean_std"))
                if full_gold is not None else "n/a"
            ),
            "full_shuffle_gold_gain_fraction": (gold_preserved if gold_preserved is not None else "n/a"),
            "js_ratio_threshold": args.h1_js_ratio_threshold,
            "gold_preserve_threshold": args.h1_gold_preserve_threshold,
        },
    )

    bin_candidates = [
        shuffle_summaries["shuffle_hnorm_bin"],
        shuffle_summaries["shuffle_pmax_bin"],
    ]
    best_bin = min(bin_candidates, key=lambda item: item["js_to_auto"])
    best_bin_gold = best_bin.get("gold_logprob_mean")
    best_bin_gold_ok = True
    if full_gold is not None and best_bin_gold is not None:
        best_bin_gold_ok = best_bin_gold >= full_gold - args.h2_gold_tolerance
    best_bin_js_improvement = float(best_bin["js_to_auto"] / max(full_js, 1e-12))
    log_hypothesis(
        logger,
        hid="H2",
        statement="Coarse uncertainty-conditioned shuffles recover at least as much structure as a fully shuffled controller, suggesting that any residual signal is mostly uncertainty-level rather than token-idiosyncratic.",
        holds=(best_bin_js_improvement <= args.h2_js_improvement_threshold and best_bin_gold_ok),
        evidence={
            "full_shuffle_js_to_auto": full_js,
            "hnorm_bin_js_to_auto": shuffle_summaries["shuffle_hnorm_bin"]["js_to_auto"],
            "pmax_bin_js_to_auto": shuffle_summaries["shuffle_pmax_bin"]["js_to_auto"],
            "best_bin_policy": str(best_bin["name"]),
            "best_bin_js_over_full_shuffle_js": best_bin_js_improvement,
            "full_shuffle_gold_logprob_covered": (full_gold if full_gold is not None else "n/a"),
            "hnorm_bin_gold_logprob_covered": (
                shuffle_summaries["shuffle_hnorm_bin"]["gold_logprob_mean"]
                if shuffle_summaries["shuffle_hnorm_bin"]["gold_logprob_mean"] is not None else "n/a"
            ),
            "pmax_bin_gold_logprob_covered": (
                shuffle_summaries["shuffle_pmax_bin"]["gold_logprob_mean"]
                if shuffle_summaries["shuffle_pmax_bin"]["gold_logprob_mean"] is not None else "n/a"
            ),
            "js_improvement_threshold": args.h2_js_improvement_threshold,
        },
    )

    log_hypothesis(
        logger,
        hid="H3",
        statement="AutoDeco's deviations from the MeanShift operating point are behaviorally small on most held-out tokens.",
        holds=(
            effect_summary["js_median"] <= args.h3_median_js_threshold
            and effect_summary["support_change_rate"] <= args.h3_support_change_threshold
            and effect_summary["nucleus_delta_gt5_rate"] <= args.h3_large_nucleus_threshold
        ),
        evidence={
            "median_js_auto_vs_mean": effect_summary["js_median"],
            "support_change_rate": effect_summary["support_change_rate"],
            "nucleus_delta_gt5_rate": effect_summary["nucleus_delta_gt5_rate"],
            "top1_absdiff_mean": effect_summary["top1_absdiff_mean"],
            "argmax_change_rate": effect_summary["argmax_change_rate"],
            "median_js_threshold": args.h3_median_js_threshold,
            "support_change_threshold": args.h3_support_change_threshold,
            "large_nucleus_threshold": args.h3_large_nucleus_threshold,
        },
    )

    summary_payload = {
        "config": {
            "path": args.path,
            "dist_k": args.dist_k,
            "val_mod": args.val_mod,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "shuffle_repeats": args.shuffle_repeats,
            "shuffle_bins": args.shuffle_bins,
            "effect_bins": args.effect_bins,
            "max_val_tokens": args.max_val_tokens,
        },
        "mean_operating_point": {
            "T_mean_train": mean_T,
            "p_mean_train": mean_p,
        },
        "policy_summaries": {
            "autodeco": auto_summary,
            "meanshift": mean_summary,
            **shuffle_summaries,
        },
        "effect_summary_auto_vs_mean": effect_summary,
        "shuffle_replicates": {
            name: [_serializable(asdict(item)) for item in items]
            for name, items in repeat_metrics.items()
        },
    }
    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_serializable(summary_payload), f, indent=2)
    logger.info("Saved metrics summary to %s", summary_path)


if __name__ == "__main__":
    main()
