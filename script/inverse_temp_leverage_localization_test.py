#!/usr/bin/env python3
"""
Localize where residual token-level control could matter.

For held-out tokens, this script bins by uncertainty signals (`H_norm`, `p_max`,
and `gap12`) and reports:
  - where curvature `v_t` is high
  - where inverse-temperature residual magnitude `|delta_t|` is high
  - where AutoDeco changes nucleus support relative to the mean operating point
  - where alignment `g_t * delta_t` is positive or negative
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from pertoken_diagnostics_hypothesis_utils import (
    compute_inverse_temp_terms,
    heldout_topk_adequacy_stats,
    load_diagnostics_slice,
    rank_bins,
    rowwise_nucleus_size_from_probs,
    softmax_from_beta,
)
from shadow_controller_dist_match import log_hypothesis, set_seed, setup_logging


def _safe_sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    mask = (np.abs(x) > 1e-12) & (np.abs(y) > 1e-12)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def _bin_table(
    signal_name: str,
    signal_values: np.ndarray,
    terms,
    abs_delta_all: np.ndarray,
    support_change: np.ndarray,
    nucleus_delta: np.ndarray,
    num_bins: int,
) -> List[Dict[str, Any]]:
    bins = rank_bins(signal_values, num_bins)
    rows: List[Dict[str, Any]] = []
    covered_mask = terms.covered_mask
    for bin_idx in range(int(bins.max()) + 1):
        all_mask = bins == bin_idx
        covered_bin_mask = covered_mask & all_mask
        row: Dict[str, Any] = {
            "signal": signal_name,
            "bin_index": int(bin_idx),
            "count_all": int(np.sum(all_mask)),
            "count_covered": int(np.sum(covered_bin_mask)),
            "signal_min": float(np.min(signal_values[all_mask])),
            "signal_max": float(np.max(signal_values[all_mask])),
            "signal_mean": float(np.mean(signal_values[all_mask])),
            "mean_abs_delta": float(np.mean(abs_delta_all[all_mask])),
            "support_change_rate": float(np.mean(support_change[all_mask])),
            "nucleus_delta_gt1_rate": float(np.mean(nucleus_delta[all_mask] > 1)),
            "nucleus_delta_gt5_rate": float(np.mean(nucleus_delta[all_mask] > 5)),
        }
        if np.any(covered_bin_mask):
            idx = np.searchsorted(terms.covered_indices, np.flatnonzero(covered_bin_mask))
            row.update(
                {
                    "mean_var": float(np.mean(terms.var[idx])),
                    "mean_alignment": float(np.mean(terms.alignment[idx])),
                    "mean_penalty": float(np.mean(terms.penalty[idx])),
                    "mean_actual_gain": float(np.mean(terms.actual_gain[idx])),
                    "positive_alignment_rate": float(np.mean(terms.alignment[idx] > 0)),
                    "sign_agreement_grad_delta": _safe_sign_agreement(terms.grad[idx], terms.delta[idx]),
                }
            )
        else:
            row.update(
                {
                    "mean_var": None,
                    "mean_alignment": None,
                    "mean_penalty": None,
                    "mean_actual_gain": None,
                    "positive_alignment_rate": None,
                    "sign_agreement_grad_delta": None,
                }
            )
        rows.append(row)
    return rows


def _extreme(rows: List[Dict[str, Any]], which: str) -> Dict[str, Any]:
    if which == "low":
        return rows[0]
    if which == "high":
        return rows[-1]
    raise ValueError(which)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to the saved per-token diagnostics DatasetDict.")
    ap.add_argument("--out-dir", type=str, default="inverse_temp_localization_runs/default")
    ap.add_argument("--dist-k", type=int, default=200)
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-tokens", type=int, default=0)
    ap.add_argument("--num-bins", type=int, default=8)
    ap.add_argument("--h0-mean-mass-threshold", type=float, default=0.97)
    ap.add_argument("--h0-frac-mass-threshold", type=float, default=0.90)
    ap.add_argument("--h0-row-mass-threshold", type=float, default=0.95)
    ap.add_argument("--h0-gold-coverage-threshold", type=float, default=0.80)
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
    logger.info(
        "Held-out rows: %d | dist_k=%d | mean_beta(train)=%.5f | mean_p(train)=%.5f",
        len(bundle.T_hat), args.dist_k, op.mean_beta, op.mean_p,
    )

    adequacy = heldout_topk_adequacy_stats(bundle, row_mass_threshold=args.h0_row_mass_threshold)
    adequacy_holds = (
        (bundle.mass_ref is None or (
            adequacy["mean_retained_mass_at_T1"] >= args.h0_mean_mass_threshold
            and adequacy[f"frac_rows_mass_ge_{args.h0_row_mass_threshold:.2f}"] >= args.h0_frac_mass_threshold
        ))
        and adequacy["gold_in_topk_rate"] >= args.h0_gold_coverage_threshold
    )
    log_hypothesis(
        logger,
        hid="H0",
        statement="The stored top-k sketch is rich enough to localize residual-control leverage on held-out tokens.",
        holds=adequacy_holds,
        evidence={
            "dist_k": args.dist_k,
            **adequacy,
            "mass_mean_threshold": args.h0_mean_mass_threshold,
            "mass_frac_threshold": args.h0_frac_mass_threshold,
            "gold_coverage_threshold": args.h0_gold_coverage_threshold,
        },
    )
    if adequacy["gold_in_topk_rate"] < args.h0_gold_coverage_threshold:
        logger.warning(
            "Gold-token coverage inside the stored top-k sketch is %.4f. If this is too low, rerun collect_pertoken_diagnostics.py with a larger --topk_sketch.",
            adequacy["gold_in_topk_rate"],
        )

    terms = compute_inverse_temp_terms(bundle, beta_bar=op.mean_beta)
    if len(terms.delta) == 0:
        raise ValueError(
            "No held-out tokens had the gold token inside the stored top-k sketch. "
            "Rerun collect_pertoken_diagnostics.py with a larger --topk_sketch."
        )
    beta_auto = 1.0 / np.clip(bundle.T_hat, 1e-6, None)
    abs_delta_all = np.abs(beta_auto - op.mean_beta)

    probs_auto = softmax_from_beta(bundle.topk_logits, beta_auto)
    probs_mean = softmax_from_beta(bundle.topk_logits, np.full(len(bundle.T_hat), op.mean_beta, dtype=np.float32))
    nucleus_auto = rowwise_nucleus_size_from_probs(probs_auto, bundle.p_hat)
    nucleus_mean = rowwise_nucleus_size_from_probs(probs_mean, op.mean_p)
    nucleus_delta = np.abs(nucleus_auto - nucleus_mean)
    support_change = nucleus_delta > 0

    bin_tables = {
        "H_norm": _bin_table("H_norm", bundle.H_norm, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
        "p_max": _bin_table("p_max", bundle.p_max, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
        "gap12": _bin_table("gap12", bundle.gap12, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
    }

    logger.info("=== Bin Tables ===")
    for signal_name, rows in bin_tables.items():
        logger.info("--- %s ---", signal_name)
        for row in rows:
            logger.info(
                "BIN %s[%d] range=[%.5f, %.5f] count=%d covered=%d | mean|delta|=%.6f | mean_v=%s | "
                "align=%s | penalty=%s | support_change=%.6f | |dN|>1=%.6f | |dN|>5=%.6f | pos_align=%s",
                signal_name,
                row["bin_index"],
                row["signal_min"],
                row["signal_max"],
                row["count_all"],
                row["count_covered"],
                row["mean_abs_delta"],
                ("n/a" if row["mean_var"] is None else f"{row['mean_var']:.6f}"),
                ("n/a" if row["mean_alignment"] is None else f"{row['mean_alignment']:.6f}"),
                ("n/a" if row["mean_penalty"] is None else f"{row['mean_penalty']:.6f}"),
                row["support_change_rate"],
                row["nucleus_delta_gt1_rate"],
                row["nucleus_delta_gt5_rate"],
                ("n/a" if row["positive_alignment_rate"] is None else f"{row['positive_alignment_rate']:.6f}"),
            )

    h_rows = bin_tables["H_norm"]
    p_rows = bin_tables["p_max"]
    h_low, h_high = _extreme(h_rows, "low"), _extreme(h_rows, "high")
    p_low, p_high = _extreme(p_rows, "low"), _extreme(p_rows, "high")

    log_hypothesis(
        logger,
        hid="H1",
        statement="Confident tokens leave little room for residual control compared with uncertain tokens.",
        holds=(
            h_high["mean_var"] is not None and h_low["mean_var"] is not None
            and h_high["mean_var"] > h_low["mean_var"]
            and h_high["support_change_rate"] > h_low["support_change_rate"]
            and p_low["mean_var"] is not None and p_high["mean_var"] is not None
            and p_low["mean_var"] > p_high["mean_var"]
        ),
        evidence={
            "H_norm_lowbin_mean_var": (h_low["mean_var"] if h_low["mean_var"] is not None else "n/a"),
            "H_norm_highbin_mean_var": (h_high["mean_var"] if h_high["mean_var"] is not None else "n/a"),
            "H_norm_lowbin_support_change": h_low["support_change_rate"],
            "H_norm_highbin_support_change": h_high["support_change_rate"],
            "p_max_lowbin_mean_var": (p_low["mean_var"] if p_low["mean_var"] is not None else "n/a"),
            "p_max_highbin_mean_var": (p_high["mean_var"] if p_high["mean_var"] is not None else "n/a"),
        },
    )

    log_hypothesis(
        logger,
        hid="H2",
        statement="Even where leverage is highest, average positive alignment is not consistently larger than the local penalty.",
        holds=(
            h_high["mean_alignment"] is not None and h_high["mean_penalty"] is not None
            and h_high["mean_alignment"] <= h_high["mean_penalty"]
            and p_low["mean_alignment"] is not None and p_low["mean_penalty"] is not None
            and p_low["mean_alignment"] <= p_low["mean_penalty"]
        ),
        evidence={
            "H_norm_highbin_mean_alignment": (h_high["mean_alignment"] if h_high["mean_alignment"] is not None else "n/a"),
            "H_norm_highbin_mean_penalty": (h_high["mean_penalty"] if h_high["mean_penalty"] is not None else "n/a"),
            "p_max_lowbin_mean_alignment": (p_low["mean_alignment"] if p_low["mean_alignment"] is not None else "n/a"),
            "p_max_lowbin_mean_penalty": (p_low["mean_penalty"] if p_low["mean_penalty"] is not None else "n/a"),
        },
    )

    payload = {
        "config": {
            "path": args.path,
            "dist_k": args.dist_k,
            "val_mod": args.val_mod,
            "seed": args.seed,
            "max_val_tokens": args.max_val_tokens,
            "num_bins": args.num_bins,
        },
        "train_operating_point": {
            "mean_T": op.mean_T,
            "mean_p": op.mean_p,
            "mean_beta": op.mean_beta,
            "T_from_mean_beta": op.T_from_mean_beta,
        },
        "adequacy": adequacy,
        "global_summary": {
            "mean_abs_delta": float(np.mean(abs_delta_all)),
            "mean_var": float(np.mean(terms.var)),
            "mean_alignment": float(np.mean(terms.alignment)),
            "mean_penalty": float(np.mean(terms.penalty)),
            "mean_actual_gain": float(np.mean(terms.actual_gain)),
            "support_change_rate": float(np.mean(support_change)),
            "nucleus_delta_gt1_rate": float(np.mean(nucleus_delta > 1)),
            "nucleus_delta_gt5_rate": float(np.mean(nucleus_delta > 5)),
        },
        "bin_tables": bin_tables,
    }
    out_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics summary to %s", out_path)


if __name__ == "__main__":
    main()
