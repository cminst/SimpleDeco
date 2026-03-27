#!/usr/bin/env python3
"""
Top-p breakpoint sparsity diagnostic.

At a fixed operating-point temperature, top-p control only matters when the
chosen `p_t` crosses a cumulative-mass breakpoint of the sorted next-token
distribution. This script measures how often that actually happens on held-out
tokens, and how large the resulting support changes are.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from pertoken_diagnostics_hypothesis_utils import (
    heldout_topk_adequacy_stats,
    load_diagnostics_slice,
    rowwise_nucleus_size_from_probs,
    softmax_from_beta,
)
from shadow_controller_dist_match import log_hypothesis, set_seed, setup_logging


def _mean_mode_temperature(op, mode: str) -> float:
    if mode == "mean_T":
        return op.mean_T
    if mode == "mean_beta":
        return op.T_from_mean_beta
    raise ValueError(mode)


def _nearest_breakpoint_distance(cdf: np.ndarray, p_bar: float) -> np.ndarray:
    return np.min(np.abs(cdf - p_bar), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to the saved per-token diagnostics DatasetDict.")
    ap.add_argument("--output-dir", dest="out_dir", type=str, default="top_p_breakpoint_runs/default")
    ap.add_argument("--dist-k", type=int, default=200)
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-tokens", type=int, default=0)
    ap.add_argument("--temperature-operating-point", choices=["mean_T", "mean_beta"], default="mean_T")
    ap.add_argument("--h0-mean-mass-threshold", type=float, default=0.97)
    ap.add_argument("--h0-frac-mass-threshold", type=float, default=0.90)
    ap.add_argument("--h0-row-mass-threshold", type=float, default=0.95)
    ap.add_argument("--h1-raw-to-support-ratio-threshold", type=float, default=2.0)
    ap.add_argument("--h2-support-change-threshold", type=float, default=0.30)
    ap.add_argument("--h2-large-change-threshold", type=float, default=0.05)
    ap.add_argument("--h3-breakpoint-ratio-threshold", type=float, default=0.75)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logging(args.out_dir)
    set_seed(args.seed)

    logger.info("Loading diagnostics dataset from %s", args.path)
    bundle, op = load_diagnostics_slice(
        ds_path=args.path,
        dist_k=args.dist_k,
        val_mod=args.val_mod,
        seed=args.seed,
        logger=logger,
        max_val_tokens=args.max_val_tokens,
    )

    T_op = _mean_mode_temperature(op, args.temperature_operating_point)
    beta_op = 1.0 / max(T_op, 1e-6)
    p_bar = op.mean_p
    logger.info(
        "Held-out rows: %d | dist_k=%d | temperature_mode=%s | T_op=%.5f | p_bar=%.5f",
        len(bundle.T_hat), args.dist_k, args.temperature_operating_point, T_op, p_bar,
    )

    adequacy = heldout_topk_adequacy_stats(bundle, row_mass_threshold=args.h0_row_mass_threshold)
    adequacy_holds = (
        bundle.mass_ref is None or (
            adequacy["mean_retained_mass_at_T1"] >= args.h0_mean_mass_threshold
            and adequacy[f"frac_rows_mass_ge_{args.h0_row_mass_threshold:.2f}"] >= args.h0_frac_mass_threshold
        )
    )
    log_hypothesis(
        logger,
        hid="H0",
        statement="The stored top-k sketch is rich enough to approximate top-p breakpoint statistics at the chosen operating-point temperature.",
        holds=adequacy_holds,
        evidence={
            "dist_k": args.dist_k,
            **adequacy,
            "mass_mean_threshold": args.h0_mean_mass_threshold,
            "mass_frac_threshold": args.h0_frac_mass_threshold,
        },
    )

    probs_op = softmax_from_beta(bundle.topk_logits, np.full(len(bundle.T_hat), beta_op, dtype=np.float32))
    cdf = np.cumsum(probs_op, axis=1)
    nucleus_mean = rowwise_nucleus_size_from_probs(probs_op, p_bar)
    nucleus_auto = rowwise_nucleus_size_from_probs(probs_op, bundle.p_hat)
    nucleus_delta = np.abs(nucleus_auto - nucleus_mean)
    support_change = nucleus_delta > 0

    delta_p = bundle.p_hat - p_bar
    abs_delta_p = np.abs(delta_p)
    nearest_break_dist = _nearest_breakpoint_distance(cdf, p_bar)
    changed_break_dist = nearest_break_dist[support_change]
    unchanged_break_dist = nearest_break_dist[~support_change]

    frac_absdp_gt_001 = float(np.mean(abs_delta_p > 0.01))
    frac_absdp_gt_005 = float(np.mean(abs_delta_p > 0.05))
    frac_absdp_gt_010 = float(np.mean(abs_delta_p > 0.10))
    support_change_rate = float(np.mean(support_change))
    nucleus_delta_gt1_rate = float(np.mean(nucleus_delta > 1))
    nucleus_delta_gt5_rate = float(np.mean(nucleus_delta > 5))
    raw_to_support_ratio = frac_absdp_gt_005 / max(support_change_rate, 1e-12)
    changed_vs_unchanged_break_ratio = float(
        np.mean(changed_break_dist) / max(np.mean(unchanged_break_dist), 1e-12)
    ) if len(changed_break_dist) > 0 and len(unchanged_break_dist) > 0 else float("nan")

    logger.info(
        "BREAKPOINTS | mean|dp|=%.6f std|dp|=%.6f | support_change=%.6f | |dN|>1=%.6f | |dN|>5=%.6f | "
        "frac|dp|>0.01=%.6f frac|dp|>0.05=%.6f frac|dp|>0.10=%.6f",
        float(np.mean(abs_delta_p)),
        float(np.std(abs_delta_p)),
        support_change_rate,
        nucleus_delta_gt1_rate,
        nucleus_delta_gt5_rate,
        frac_absdp_gt_001,
        frac_absdp_gt_005,
        frac_absdp_gt_010,
    )
    logger.info(
        "BREAKPOINT_DISTANCE | mean_changed=%.6f | mean_unchanged=%.6f | changed/unchanged=%.6f",
        (float(np.mean(changed_break_dist)) if len(changed_break_dist) > 0 else float("nan")),
        (float(np.mean(unchanged_break_dist)) if len(unchanged_break_dist) > 0 else float("nan")),
        changed_vs_unchanged_break_ratio,
    )

    log_hypothesis(
        logger,
        hid="H1",
        statement="Raw top-p variation is much more common than actual nucleus-support changes at a fixed operating-point temperature.",
        holds=(raw_to_support_ratio >= args.h1_raw_to_support_ratio_threshold),
        evidence={
            "frac_abs_delta_p_gt_0.05": frac_absdp_gt_005,
            "support_change_rate": support_change_rate,
            "raw_to_support_ratio": raw_to_support_ratio,
            "ratio_threshold": args.h1_raw_to_support_ratio_threshold,
        },
    )

    log_hypothesis(
        logger,
        hid="H2",
        statement="When top-p does change the nucleus at the operating-point temperature, the effective support shifts are sparse and usually small.",
        holds=(
            support_change_rate <= args.h2_support_change_threshold
            and nucleus_delta_gt5_rate <= args.h2_large_change_threshold
        ),
        evidence={
            "support_change_rate": support_change_rate,
            "nucleus_delta_gt1_rate": nucleus_delta_gt1_rate,
            "nucleus_delta_gt5_rate": nucleus_delta_gt5_rate,
            "support_change_threshold": args.h2_support_change_threshold,
            "large_change_threshold": args.h2_large_change_threshold,
        },
    )

    log_hypothesis(
        logger,
        hid="H3",
        statement="The tokens whose nucleus changes are closer to a top-p breakpoint than the tokens whose nucleus does not change.",
        holds=(
            np.isfinite(changed_vs_unchanged_break_ratio)
            and changed_vs_unchanged_break_ratio <= args.h3_breakpoint_ratio_threshold
        ),
        evidence={
            "mean_breakpoint_distance_changed": (
                float(np.mean(changed_break_dist)) if len(changed_break_dist) > 0 else "n/a"
            ),
            "mean_breakpoint_distance_unchanged": (
                float(np.mean(unchanged_break_dist)) if len(unchanged_break_dist) > 0 else "n/a"
            ),
            "changed_vs_unchanged_ratio": changed_vs_unchanged_break_ratio,
            "ratio_threshold": args.h3_breakpoint_ratio_threshold,
        },
    )

    payload: Dict[str, Any] = {
        "config": {
            "path": args.path,
            "dist_k": args.dist_k,
            "val_mod": args.val_mod,
            "seed": args.seed,
            "max_val_tokens": args.max_val_tokens,
            "temperature_operating_point": args.temperature_operating_point,
        },
        "train_operating_point": {
            "mean_T": op.mean_T,
            "mean_p": op.mean_p,
            "mean_beta": op.mean_beta,
            "T_from_mean_beta": op.T_from_mean_beta,
            "T_op": T_op,
            "beta_op": beta_op,
        },
        "adequacy": adequacy,
        "summary": {
            "mean_abs_delta_p": float(np.mean(abs_delta_p)),
            "std_abs_delta_p": float(np.std(abs_delta_p)),
            "support_change_rate": support_change_rate,
            "nucleus_delta_gt1_rate": nucleus_delta_gt1_rate,
            "nucleus_delta_gt5_rate": nucleus_delta_gt5_rate,
            "frac_abs_delta_p_gt_0.01": frac_absdp_gt_001,
            "frac_abs_delta_p_gt_0.05": frac_absdp_gt_005,
            "frac_abs_delta_p_gt_0.10": frac_absdp_gt_010,
            "raw_to_support_ratio": raw_to_support_ratio,
            "mean_breakpoint_distance_changed": (
                float(np.mean(changed_break_dist)) if len(changed_break_dist) > 0 else None
            ),
            "mean_breakpoint_distance_unchanged": (
                float(np.mean(unchanged_break_dist)) if len(unchanged_break_dist) > 0 else None
            ),
            "changed_vs_unchanged_breakpoint_ratio": changed_vs_unchanged_break_ratio,
        },
    }
    out_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics summary to %s", out_path)


if __name__ == "__main__":
    main()
