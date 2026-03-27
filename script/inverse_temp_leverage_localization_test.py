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


DEFAULT_ENTROPY_BIN_EDGES = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def _safe_sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    mask = (np.abs(x) > 1e-12) & (np.abs(y) > 1e-12)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def _parse_bin_edges(spec: str) -> np.ndarray:
    try:
        edges = np.asarray([float(part.strip()) for part in spec.split(",") if part.strip()], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Failed to parse --entropy-bin-edges={spec!r}") from exc
    if len(edges) < 2:
        raise ValueError("Expected at least two comma-separated entropy bin edges.")
    if not np.all(np.isfinite(edges)):
        raise ValueError("Entropy bin edges must be finite.")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("Entropy bin edges must be strictly increasing.")
    return edges


def _format_bin_label(left: float, right: float, is_last: bool) -> str:
    return f"$[{left:.2f}, {right:.2f}{']' if is_last else ')'}$"


def _fixed_interval_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, float(edges[0]), float(edges[-1]))
    bins = np.searchsorted(edges, clipped, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2).astype(np.int32)


def _covered_lookup(terms) -> np.ndarray:
    lookup = np.full(len(terms.covered_mask), -1, dtype=np.int32)
    lookup[terms.covered_indices] = np.arange(len(terms.covered_indices), dtype=np.int32)
    return lookup


def _format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_latex_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{digits}f}"


def _format_latex_percent(count: int, total: int, digits: int = 1) -> str:
    if total <= 0:
        return "---"
    return f"{100.0 * float(count) / float(total):.{digits}f}\\%"


def _latex_row(row: Dict[str, Any], use_covered_count: bool, total_count: int) -> str:
    count = row["count_covered"] if use_covered_count else row["count_all"]
    support = row["support_change_rate_covered"] if use_covered_count else row["support_change_rate"]
    return (
        f"{row['bin_label']} & {_format_latex_percent(count, total_count)} & "
        f"{_format_latex_float(row['mean_alignment'])} & "
        f"{_format_latex_float(row['mean_penalty'])} & "
        f"{_format_latex_float(row['mean_predicted_net'])} & "
        f"{_format_latex_float(row['mean_logit_variance'])} & "
        f"{_format_latex_float(support)} \\\\"
    )


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
    covered_lookup = _covered_lookup(terms)
    for bin_idx in range(int(bins.max()) + 1):
        all_mask = bins == bin_idx
        covered_bin_mask = covered_mask & all_mask
        row: Dict[str, Any] = {
            "signal": signal_name,
            "bin_index": int(bin_idx),
            "count_all": int(np.sum(all_mask)),
            "count_covered": int(np.sum(covered_bin_mask)),
            "covered_rate": (float(np.sum(covered_bin_mask) / np.sum(all_mask)) if np.any(all_mask) else None),
            "signal_min": float(np.min(signal_values[all_mask])),
            "signal_max": float(np.max(signal_values[all_mask])),
            "signal_mean": float(np.mean(signal_values[all_mask])),
            "mean_abs_delta": float(np.mean(abs_delta_all[all_mask])),
            "support_change_rate": float(np.mean(support_change[all_mask])),
            "nucleus_delta_gt1_rate": float(np.mean(nucleus_delta[all_mask] > 1)),
            "nucleus_delta_gt5_rate": float(np.mean(nucleus_delta[all_mask] > 5)),
        }
        if np.any(covered_bin_mask):
            idx = covered_lookup[np.flatnonzero(covered_bin_mask)]
            row.update(
                {
                    "mean_var": float(np.mean(terms.var[idx])),
                    "mean_logit_variance": float(np.mean(terms.var[idx])),
                    "delta_variance": float(np.var(terms.delta[idx])),
                    "mean_alignment": float(np.mean(terms.alignment[idx])),
                    "mean_penalty": float(np.mean(terms.penalty[idx])),
                    "mean_predicted_net": float(np.mean(terms.predicted_net[idx])),
                    "mean_actual_gain": float(np.mean(terms.actual_gain[idx])),
                    "positive_alignment_rate": float(np.mean(terms.alignment[idx] > 0)),
                    "sign_agreement_grad_delta": _safe_sign_agreement(terms.grad[idx], terms.delta[idx]),
                    "mean_abs_delta_covered": float(np.mean(np.abs(terms.delta[idx]))),
                    "support_change_rate_covered": float(np.mean(support_change[covered_bin_mask])),
                    "nucleus_delta_gt1_rate_covered": float(np.mean(nucleus_delta[covered_bin_mask] > 1)),
                    "nucleus_delta_gt5_rate_covered": float(np.mean(nucleus_delta[covered_bin_mask] > 5)),
                }
            )
        else:
            row.update(
                {
                    "mean_var": None,
                    "mean_logit_variance": None,
                    "delta_variance": None,
                    "mean_alignment": None,
                    "mean_penalty": None,
                    "mean_predicted_net": None,
                    "mean_actual_gain": None,
                    "positive_alignment_rate": None,
                    "sign_agreement_grad_delta": None,
                    "mean_abs_delta_covered": None,
                    "support_change_rate_covered": None,
                    "nucleus_delta_gt1_rate_covered": None,
                    "nucleus_delta_gt5_rate_covered": None,
                }
            )
        rows.append(row)
    return rows


def _fixed_entropy_table(
    entropy_values: np.ndarray,
    edges: np.ndarray,
    terms,
    abs_delta_all: np.ndarray,
    support_change: np.ndarray,
    nucleus_delta: np.ndarray,
) -> List[Dict[str, Any]]:
    bins = _fixed_interval_bins(entropy_values, edges)
    rows: List[Dict[str, Any]] = []
    covered_mask = terms.covered_mask
    covered_lookup = _covered_lookup(terms)
    for bin_idx in range(len(edges) - 1):
        left = float(edges[bin_idx])
        right = float(edges[bin_idx + 1])
        all_mask = bins == bin_idx
        covered_bin_mask = covered_mask & all_mask
        row: Dict[str, Any] = {
            "signal": "H_norm_fixed",
            "bin_index": int(bin_idx),
            "bin_left": left,
            "bin_right": right,
            "bin_label": _format_bin_label(left, right, is_last=(bin_idx == len(edges) - 2)),
            "count_all": int(np.sum(all_mask)),
            "count_covered": int(np.sum(covered_bin_mask)),
            "covered_rate": (float(np.sum(covered_bin_mask) / np.sum(all_mask)) if np.any(all_mask) else None),
            "signal_min": left,
            "signal_max": right,
            "signal_mean": (float(np.mean(entropy_values[all_mask])) if np.any(all_mask) else None),
            "mean_abs_delta": (float(np.mean(abs_delta_all[all_mask])) if np.any(all_mask) else None),
            "support_change_rate": (float(np.mean(support_change[all_mask])) if np.any(all_mask) else None),
            "nucleus_delta_gt1_rate": (float(np.mean(nucleus_delta[all_mask] > 1)) if np.any(all_mask) else None),
            "nucleus_delta_gt5_rate": (float(np.mean(nucleus_delta[all_mask] > 5)) if np.any(all_mask) else None),
        }
        if np.any(covered_bin_mask):
            idx = covered_lookup[np.flatnonzero(covered_bin_mask)]
            row.update(
                {
                    "mean_var": float(np.mean(terms.var[idx])),
                    "mean_logit_variance": float(np.mean(terms.var[idx])),
                    "delta_variance": float(np.var(terms.delta[idx])),
                    "mean_alignment": float(np.mean(terms.alignment[idx])),
                    "mean_penalty": float(np.mean(terms.penalty[idx])),
                    "mean_predicted_net": float(np.mean(terms.predicted_net[idx])),
                    "mean_actual_gain": float(np.mean(terms.actual_gain[idx])),
                    "positive_alignment_rate": float(np.mean(terms.alignment[idx] > 0)),
                    "sign_agreement_grad_delta": _safe_sign_agreement(terms.grad[idx], terms.delta[idx]),
                    "mean_abs_delta_covered": float(np.mean(np.abs(terms.delta[idx]))),
                    "support_change_rate_covered": float(np.mean(support_change[covered_bin_mask])),
                    "nucleus_delta_gt1_rate_covered": float(np.mean(nucleus_delta[covered_bin_mask] > 1)),
                    "nucleus_delta_gt5_rate_covered": float(np.mean(nucleus_delta[covered_bin_mask] > 5)),
                }
            )
        else:
            row.update(
                {
                    "mean_var": None,
                    "mean_logit_variance": None,
                    "delta_variance": None,
                    "mean_alignment": None,
                    "mean_penalty": None,
                    "mean_predicted_net": None,
                    "mean_actual_gain": None,
                    "positive_alignment_rate": None,
                    "sign_agreement_grad_delta": None,
                    "mean_abs_delta_covered": None,
                    "support_change_rate_covered": None,
                    "nucleus_delta_gt1_rate_covered": None,
                    "nucleus_delta_gt5_rate_covered": None,
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
    ap.add_argument("--output-dir", dest="out_dir", type=str, default="inverse_temp_localization_runs/default")
    ap.add_argument("--dist-k", type=int, default=200)
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-tokens", type=int, default=0)
    ap.add_argument("--num-bins", type=int, default=8)
    ap.add_argument(
        "--use-all-tokens",
        action="store_true",
        help="Estimate the mean operating point and report localization metrics on the full diagnostics dataset.",
    )
    ap.add_argument(
        "--entropy-bin-edges",
        type=str,
        default=",".join(f"{edge:.2f}" for edge in DEFAULT_ENTROPY_BIN_EDGES),
        help="Comma-separated fixed H_norm bin edges for the paper-facing entropy table.",
    )
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
        use_all_tokens=args.use_all_tokens,
    )
    logger.info(
        "%s rows: %d | dist_k=%d | mean_beta(op)=%.5f | mean_p(op)=%.5f",
        ("All-token evaluation" if args.use_all_tokens else "Held-out"),
        len(bundle.T_hat),
        args.dist_k,
        op.mean_beta,
        op.mean_p,
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
    entropy_edges = _parse_bin_edges(args.entropy_bin_edges)

    bin_tables = {
        "H_norm": _bin_table("H_norm", bundle.H_norm, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
        "p_max": _bin_table("p_max", bundle.p_max, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
        "gap12": _bin_table("gap12", bundle.gap12, terms, abs_delta_all, support_change, nucleus_delta, args.num_bins),
    }
    entropy_table = _fixed_entropy_table(
        bundle.H_norm,
        entropy_edges,
        terms,
        abs_delta_all,
        support_change,
        nucleus_delta,
    )

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

    logger.info("=== Fixed Entropy Table (paper bins) ===")
    for row in entropy_table:
        logger.info(
            "ENTROPY %s | N_all=%d | N_cov=%d | cov=%.6f | align=%s | penalty=%s | pred_net=%s | "
            "actual_net=%s | mean_v=%s | var_delta=%s | support_change(all)=%s | support_change(cov)=%s",
            row["bin_label"],
            row["count_all"],
            row["count_covered"],
            (float("nan") if row["covered_rate"] is None else row["covered_rate"]),
            _format_metric(row["mean_alignment"]),
            _format_metric(row["mean_penalty"]),
            _format_metric(row["mean_predicted_net"]),
            _format_metric(row["mean_actual_gain"]),
            _format_metric(row["mean_logit_variance"]),
            _format_metric(row["delta_variance"]),
            _format_metric(row["support_change_rate"]),
            _format_metric(row["support_change_rate_covered"]),
        )

    total_count_covered = int(sum(row["count_covered"] for row in entropy_table))
    total_count_all = int(sum(row["count_all"] for row in entropy_table))
    latex_rows_covered = [
        _latex_row(row, use_covered_count=True, total_count=total_count_covered) for row in entropy_table
    ]
    latex_rows_all = [
        _latex_row(row, use_covered_count=False, total_count=total_count_all) for row in entropy_table
    ]
    latex_cov_path = os.path.join(args.out_dir, "entropy_bin_table_rows_count_covered.tex")
    latex_all_path = os.path.join(args.out_dir, "entropy_bin_table_rows_count_all.tex")
    with open(latex_cov_path, "w") as f:
        f.write(
            "% Ready-to-copy rows for colm2026_v5.tex.\n"
            "% N is the percentage share of covered tokens, and Support Delta rate uses the same covered subset.\n"
            "% Net gain here is the predicted net gain E[g*delta - 0.5*v*delta^2].\n"
            "% The JSON summary also includes actual net gain and count_all if you prefer those.\n"
        )
        f.write("\n".join(latex_rows_covered))
        f.write("\n")
    with open(latex_all_path, "w") as f:
        f.write(
            "% Alternate rows where N is the percentage share of all evaluation tokens,\n"
            "% and Support Delta rate uses all evaluation tokens.\n"
            "% Alignment/penalty/net/v_t are still computed on covered tokens only.\n"
        )
        f.write("\n".join(latex_rows_all))
        f.write("\n")
    logger.info("Saved covered-count LaTeX rows to %s", latex_cov_path)
    logger.info("Saved all-token-count LaTeX rows to %s", latex_all_path)

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
            "entropy_bin_edges": [float(edge) for edge in entropy_edges],
            "use_all_tokens": args.use_all_tokens,
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
        "entropy_table": entropy_table,
        "latex_rows": {
            "count_covered": latex_rows_covered,
            "count_all": latex_rows_all,
        },
    }
    out_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics summary to %s", out_path)


if __name__ == "__main__":
    main()
