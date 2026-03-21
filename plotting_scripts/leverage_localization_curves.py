"""Generate the paper-style leverage localization figure."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _REPO_ROOT / "script"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import compare_trajs as ct


DEFAULT_OUTPUT = _REPO_ROOT / "figure" / "leverage_localization_curves.pdf"
DEFAULT_MERGE_TAIL_FROM = 0.40


def _require_mapping(payload: Any, desc: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {desc} to be a JSON object.")
    return payload


def _require_number(row: dict[str, Any], key: str, row_idx: int) -> float:
    value = row.get(key)
    if value is None or not isinstance(value, (int, float)):
        raise ValueError(f"Row {row_idx} is missing numeric field '{key}'.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Row {row_idx} field '{key}' must be finite.")
    return numeric


def _optional_number(row: dict[str, Any], key: str, row_idx: int) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Row {row_idx} field '{key}' must be numeric or null.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Row {row_idx} field '{key}' must be finite or null.")
    return numeric


def _load_entropy_table(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r") as f:
        payload = json.load(f)
    payload = _require_mapping(payload, "top-level payload")
    rows = payload.get("entropy_table")
    if not isinstance(rows, list):
        raise ValueError("metrics_summary.json must contain an 'entropy_table' list.")
    if not rows:
        raise ValueError("'entropy_table' is empty; nothing to plot.")
    return rows


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _format_edge(value: float) -> str:
    rounded = round(float(value) + 1e-9, 2)
    if abs(rounded) < 1e-9:
        return "0"
    if abs(rounded - 1.0) < 1e-9:
        return "1"
    text = f"{rounded:.2f}"
    if 0.0 < rounded < 1.0 and text.startswith("0"):
        return text[1:]
    if -1.0 < rounded < 0.0 and text.startswith("-0"):
        return f"-{text[2:]}"
    return text


def _interval_label(left: float, right: float) -> str:
    close = "]" if abs(right - 1.0) < 1e-9 else ")"
    return rf"$[{_format_edge(left)},{_format_edge(right)}{close}$"


def _tail_label(left: float) -> str:
    return rf"$\geq {_format_edge(left)}$"


def _prepare_plot_payload(
    rows: list[dict[str, Any]],
    merge_tail_from: float | None,
) -> dict[str, np.ndarray | list[str]]:
    parsed_rows: list[dict[str, float]] = []
    for row_idx, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"entropy_table[{row_idx}]")
        left = _require_number(row, "bin_left", row_idx)
        right = _require_number(row, "bin_right", row_idx)
        if right <= left:
            raise ValueError(
                f"Row {row_idx} has invalid entropy bin bounds: bin_right <= bin_left."
            )
        parsed_rows.append(
            {
                "left": left,
                "right": right,
                "count_covered": _require_number(row, "count_covered", row_idx),
                "alignment": _optional_number(row, "mean_alignment", row_idx),
                "penalty": _optional_number(row, "mean_penalty", row_idx),
                "support_change": (
                    _optional_number(row, "support_change_rate_covered", row_idx)
                    if row.get("support_change_rate_covered") is not None
                    else _optional_number(row, "support_change_rate", row_idx)
                ),
            }
        )

    left_arr = np.asarray([row["left"] for row in parsed_rows], dtype=np.float64)
    right_arr = np.asarray([row["right"] for row in parsed_rows], dtype=np.float64)
    if np.any(left_arr[1:] < left_arr[:-1]) or np.any(right_arr[1:] < right_arr[:-1]):
        raise ValueError("Entropy bins must appear in nondecreasing order.")

    merged_rows: list[dict[str, float | str]] = []
    tail_rows: list[dict[str, float]] = []
    for row in parsed_rows:
        if merge_tail_from is not None and row["left"] >= merge_tail_from:
            tail_rows.append(row)
            continue
        merged_rows.append({**row, "label": _interval_label(row["left"], row["right"])})

    if tail_rows:
        weights = np.asarray([row["count_covered"] for row in tail_rows], dtype=np.float64)
        merged_rows.append(
            {
                "left": float(tail_rows[0]["left"]),
                "right": float(tail_rows[-1]["right"]),
                "count_covered": float(np.sum(weights)),
                "alignment": _weighted_mean(
                    np.asarray([row["alignment"] for row in tail_rows], dtype=np.float64),
                    weights,
                ),
                "penalty": _weighted_mean(
                    np.asarray([row["penalty"] for row in tail_rows], dtype=np.float64),
                    weights,
                ),
                "support_change": _weighted_mean(
                    np.asarray([row["support_change"] for row in tail_rows], dtype=np.float64),
                    weights,
                ),
                "label": _tail_label(float(tail_rows[0]["left"])),
            }
        )

    if not merged_rows:
        raise ValueError("No entropy bins remain after preprocessing.")

    counts = np.asarray([float(row["count_covered"]) for row in merged_rows], dtype=np.float64)
    total_covered = float(np.sum(counts))
    if total_covered <= 0.0:
        raise ValueError("Covered-token total is zero; cannot render the share panel.")

    alignment = np.asarray([float(row["alignment"]) for row in merged_rows], dtype=np.float64)
    penalty = np.asarray([float(row["penalty"]) for row in merged_rows], dtype=np.float64)
    support_change = np.asarray(
        [float(row["support_change"]) for row in merged_rows], dtype=np.float64
    )
    finite_any = np.isfinite(alignment) | np.isfinite(penalty)
    if not np.any(finite_any):
        raise ValueError("No finite mean_alignment or mean_penalty values were found.")
    if not np.any(np.isfinite(support_change)):
        raise ValueError(
            "No finite support_change_rate_covered or support_change_rate values were found."
        )

    return {
        "labels": [str(row["label"]) for row in merged_rows],
        "alignment": alignment,
        "penalty": penalty,
        "support_change_pct": 100.0 * support_change,
        "covered_share_pct": 100.0 * counts / total_covered,
    }


def _style_axes(ax: Any) -> None:
    ct._apply_paper_axes_style(ax)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#98A2B3")
        ax.spines[spine_name].set_linewidth(0.9)


def _plot_binned_figure(
    output_path: Path,
    plot_payload: dict[str, np.ndarray | list[str]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to generate this figure.") from exc

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{mathpazo}",
            "font.family": "serif",
            "font.serif": ["Palatino", "Times New Roman", "Times"],
            "font.size": 9.4,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.4,
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    labels = list(plot_payload["labels"])
    alignment = np.asarray(plot_payload["alignment"], dtype=np.float64)
    penalty = np.asarray(plot_payload["penalty"], dtype=np.float64)
    support_change_pct = np.asarray(plot_payload["support_change_pct"], dtype=np.float64)
    share_pct = np.asarray(plot_payload["covered_share_pct"], dtype=np.float64)

    x = np.arange(len(labels), dtype=np.float64)
    bar_width = 0.34
    blue = "#4E79A7"
    red = "#E15759"

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.45, 3.15),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.0, 0.88)},
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.31, top=0.855, wspace=0.18)

    _style_axes(ax_left)
    _style_axes(ax_right)

    align_bars = ax_left.bar(
        x - bar_width / 2.0,
        alignment,
        width=bar_width,
        color=blue,
        edgecolor="#FFFFFF",
        linewidth=0.65,
        alpha=0.95,
        zorder=3,
        label="Alignment gain",
    )
    penalty_bars = ax_left.bar(
        x + bar_width / 2.0,
        penalty,
        width=bar_width,
        color=red,
        edgecolor="#FFFFFF",
        linewidth=0.65,
        alpha=0.95,
        zorder=3,
        label="Curvature penalty",
    )
    support_bars = ax_right.bar(
        x,
        support_change_pct,
        width=0.56,
        color="#8FA4BF",
        edgecolor="#FFFFFF",
        linewidth=0.65,
        alpha=0.92,
        zorder=3,
        label="Support change",
    )

    for axis in (ax_left, ax_right):
        axis.axhline(0.0, color="#6B7280", linewidth=1.15, zorder=1)
        axis.grid(axis="y", color="#E7ECF2", linewidth=0.55, alpha=0.9)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_xlabel("normalized entropy bin")

    ax_left.set_ylabel("mean term")
    ax_right.set_ylabel(r"support change (\%)")

    left_values = np.concatenate(
        [
            alignment[np.isfinite(alignment)],
            penalty[np.isfinite(penalty)],
        ]
    )
    left_max = max(0.0, float(np.max(left_values)) if left_values.size else 1.0)
    left_bottom = -max(left_max * 0.05, 5e-4)
    ax_left.set_ylim(left_bottom, left_max * 1.12 if left_max > 0.0 else 1.0)

    right_values = support_change_pct[np.isfinite(support_change_pct)]
    right_max = max(0.0, float(np.max(right_values)) if right_values.size else 1.0)
    right_bottom = -max(right_max * 0.05, 0.2)
    ax_right.set_ylim(right_bottom, right_max * 1.12 if right_max > 0.0 else 1.0)

    tick_labels = [
        rf"{label}" + "\n" + rf"{share:.1f}\%"
        for label, share in zip(labels, share_pct)
    ]
    rotation = 12 if len(labels) > 5 else 0
    for axis in (ax_left, ax_right):
        axis.set_xticks(x)
        axis.set_xticklabels(
            tick_labels,
            rotation=rotation,
            ha="right" if rotation else "center",
        )

    legend_handles = [align_bars, penalty_bars, support_bars]
    legend_labels = ["Alignment gain", "Curvature penalty", "Support change"]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.958),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#D7DCE3",
        facecolor="#FFFFFF",
        fontsize=8.5,
        handlelength=1.9,
        handletextpad=0.55,
        borderpad=0.35,
        columnspacing=1.15,
    )
    legend.get_frame().set_linewidth(0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the leverage localization paper figure."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a metrics_summary.json file from inverse_temp_leverage_localization_test.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PDF/PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--merge-tail-from",
        type=float,
        default=DEFAULT_MERGE_TAIL_FROM,
        help=(
            "Merge all bins with bin_left >= this threshold into one tail bin "
            f"(default: {DEFAULT_MERGE_TAIL_FROM:.2f})."
        ),
    )
    parser.add_argument(
        "--no-merge-tail",
        action="store_true",
        help="Disable tail-bin merging and plot all bins separately.",
    )
    parser.add_argument(
        "--hide-net-marker",
        action="store_true",
        help="Deprecated; net markers are no longer plotted.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input metrics summary not found: {input_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    merge_tail_from = None if args.no_merge_tail else float(args.merge_tail_from)
    rows = _load_entropy_table(input_path)
    plot_payload = _prepare_plot_payload(rows, merge_tail_from=merge_tail_from)
    _plot_binned_figure(output_path, plot_payload=plot_payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
