#!/usr/bin/env python3
"""
Teacher-forced inverse-temperature alignment diagnostic.

This script tests the decomposition

    Delta ell_t ~= g_t * delta_t - 0.5 * v_t * delta_t^2

at a held-out mean operating point in inverse-temperature space, where
    beta_t = 1 / T_hat_t
    beta_bar = E_train[beta_t]
    delta_t = beta_t - beta_bar
    g_t = d ell_t / d beta at beta_bar
    v_t = - d^2 ell_t / d beta^2 at beta_bar

using the stored top-k teacher-forced diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from pertoken_diagnostics_hypothesis_utils import (
    compute_inverse_temp_terms,
    heldout_topk_adequacy_stats,
    load_diagnostics_slice,
)
from shadow_controller_dist_match import log_hypothesis, safe_div, set_seed, setup_logging


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def _safe_sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    mask = (np.abs(x) > 1e-12) & (np.abs(y) > 1e-12)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def _summarize_terms(terms) -> Dict[str, Any]:
    alignment_mean = float(np.mean(terms.alignment))
    penalty_mean = float(np.mean(terms.penalty))
    predicted_net_mean = float(np.mean(terms.predicted_net))
    actual_gain_mean = float(np.mean(terms.actual_gain))
    alignment_efficiency = safe_div(
        float(np.mean(terms.alignment)),
        float(np.sqrt(np.mean(np.square(terms.grad)) * np.mean(np.square(terms.delta)))),
    )
    oracle_gain = np.divide(
        np.square(terms.grad),
        2.0 * terms.var,
        out=np.full_like(terms.grad, np.nan),
        where=terms.var > 1e-12,
    )
    oracle_gain_valid = np.isfinite(oracle_gain)
    oracle_gain_mean = (
        float(np.mean(oracle_gain[oracle_gain_valid]))
        if np.any(oracle_gain_valid)
        else float("nan")
    )
    summary = {
        "covered_token_count": int(len(terms.delta)),
        "delta_mean": float(np.mean(terms.delta)),
        "delta_abs_mean": float(np.mean(np.abs(terms.delta))),
        "delta_abs_p90": float(np.quantile(np.abs(terms.delta), 0.90)),
        "grad_mean": float(np.mean(terms.grad)),
        "grad_abs_mean": float(np.mean(np.abs(terms.grad))),
        "var_mean": float(np.mean(terms.var)),
        "alignment_gain_mean": alignment_mean,
        "variance_penalty_mean": penalty_mean,
        "predicted_net_mean": predicted_net_mean,
        "actual_gain_mean": actual_gain_mean,
        "alignment_minus_penalty": float(alignment_mean - penalty_mean),
        "corr_grad_delta": _corr(terms.grad, terms.delta),
        "sign_agreement_grad_delta": _safe_sign_agreement(terms.grad, terms.delta),
        "alignment_efficiency": float(alignment_efficiency),
        "oracle_gain_mean": oracle_gain_mean,
        "oracle_gain_valid_rate": float(np.mean(oracle_gain_valid)),
        "oracle_efficiency": float(safe_div(actual_gain_mean, oracle_gain_mean)),
        "predicted_oracle_efficiency": float(safe_div(predicted_net_mean, oracle_gain_mean)),
        "corr_predicted_actual_gain": _corr(terms.predicted_net, terms.actual_gain),
        "corr_alignment_actual_gain": _corr(terms.alignment, terms.actual_gain),
        "actual_minus_predicted_mae": float(np.mean(np.abs(terms.actual_gain - terms.predicted_net))),
        "positive_actual_gain_rate": float(np.mean(terms.actual_gain > 0)),
        "positive_alignment_rate": float(np.mean(terms.alignment > 0)),
        "penalty_dominates_rate": float(np.mean(terms.penalty >= np.maximum(terms.alignment, 0.0))),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path to the saved per-token diagnostics DatasetDict.")
    ap.add_argument("--output-dir", dest="out_dir", type=str, default="inverse_temp_alignment_runs/default")
    ap.add_argument("--dist-k", type=int, default=200)
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-tokens", type=int, default=0, help="Optional cap on held-out tokens (0 = all).")
    ap.add_argument("--h0-mean-mass-threshold", type=float, default=0.97)
    ap.add_argument("--h0-frac-mass-threshold", type=float, default=0.90)
    ap.add_argument("--h0-row-mass-threshold", type=float, default=0.95)
    ap.add_argument("--h0-gold-coverage-threshold", type=float, default=0.80)
    ap.add_argument("--h1-efficiency-threshold", type=float, default=0.25)
    ap.add_argument("--h2-taylor-corr-threshold", type=float, default=0.50)
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
        "Held-out rows: %d | dist_k=%d | mean_T(train)=%.5f | mean_beta(train)=%.5f | T_from_mean_beta=%.5f",
        len(bundle.T_hat), args.dist_k, op.mean_T, op.mean_beta, op.T_from_mean_beta,
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
        statement="The stored top-k sketch is rich enough to approximate inverse-temperature teacher-forced diagnostics on held-out tokens.",
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
    summary = _summarize_terms(terms)
    logger.info(
        "ALIGNMENT | covered=%d | E[g*delta]=%.6f | 0.5E[v*delta^2]=%.6f | predicted_net=%.6f | actual_gain=%.6f | "
        "corr(g,delta)=%.6f | sign_agree=%.6f | align_eff=%.6f",
        summary["covered_token_count"],
        summary["alignment_gain_mean"],
        summary["variance_penalty_mean"],
        summary["predicted_net_mean"],
        summary["actual_gain_mean"],
        summary["corr_grad_delta"],
        summary["sign_agreement_grad_delta"],
        summary["alignment_efficiency"],
    )
    logger.info(
        "ORACLE | oracle_gain=%.6f | oracle_valid_rate=%.6f | oracle_eff(actual)=%.6f | oracle_eff(pred)=%.6f",
        summary["oracle_gain_mean"],
        summary["oracle_gain_valid_rate"],
        summary["oracle_efficiency"],
        summary["predicted_oracle_efficiency"],
    )
    logger.info(
        "TAYLOR | corr(predicted,actual)=%.6f | corr(alignment,actual)=%.6f | mae(actual-predicted)=%.6f | "
        "positive_actual_gain_rate=%.6f | penalty_dominates_rate=%.6f",
        summary["corr_predicted_actual_gain"],
        summary["corr_alignment_actual_gain"],
        summary["actual_minus_predicted_mae"],
        summary["positive_actual_gain_rate"],
        summary["penalty_dominates_rate"],
    )

    log_hypothesis(
        logger,
        hid="H1",
        statement="Residual inverse-temperature variation is not aligned strongly enough to overcome the concavity penalty at the held-out mean-beta operating point.",
        holds=(
            summary["alignment_gain_mean"] <= summary["variance_penalty_mean"]
            and summary["actual_gain_mean"] <= 0.0
            and summary["alignment_efficiency"] <= args.h1_efficiency_threshold
        ),
        evidence={
            "alignment_gain_mean": summary["alignment_gain_mean"],
            "variance_penalty_mean": summary["variance_penalty_mean"],
            "predicted_net_mean": summary["predicted_net_mean"],
            "actual_gain_mean": summary["actual_gain_mean"],
            "corr_grad_delta": summary["corr_grad_delta"],
            "sign_agreement_grad_delta": summary["sign_agreement_grad_delta"],
            "alignment_efficiency": summary["alignment_efficiency"],
            "alignment_efficiency_threshold": args.h1_efficiency_threshold,
            "oracle_gain_mean": summary["oracle_gain_mean"],
            "oracle_efficiency": summary["oracle_efficiency"],
            "predicted_oracle_efficiency": summary["predicted_oracle_efficiency"],
        },
    )

    log_hypothesis(
        logger,
        hid="H2",
        statement="The local alignment-minus-penalty decomposition tracks the observed temperature-only gold-logprob change directionally on held-out tokens.",
        holds=(
            np.isfinite(summary["corr_predicted_actual_gain"])
            and summary["corr_predicted_actual_gain"] >= args.h2_taylor_corr_threshold
        ),
        evidence={
            "corr_predicted_actual_gain": summary["corr_predicted_actual_gain"],
            "corr_alignment_actual_gain": summary["corr_alignment_actual_gain"],
            "actual_minus_predicted_mae": summary["actual_minus_predicted_mae"],
            "taylor_corr_threshold": args.h2_taylor_corr_threshold,
        },
    )

    payload = {
        "config": {
            "path": args.path,
            "dist_k": args.dist_k,
            "val_mod": args.val_mod,
            "seed": args.seed,
            "max_val_tokens": args.max_val_tokens,
        },
        "train_operating_point": {
            "mean_T": op.mean_T,
            "mean_p": op.mean_p,
            "mean_beta": op.mean_beta,
            "T_from_mean_beta": op.T_from_mean_beta,
        },
        "adequacy": adequacy,
        "summary": summary,
    }
    out_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics summary to %s", out_path)


if __name__ == "__main__":
    main()
