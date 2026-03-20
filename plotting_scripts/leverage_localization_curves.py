"""Generate the paper-style leverage localization curve figure."""
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


def _prepare_plot_payload(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    lefts: list[float] = []
    rights: list[float] = []
    centers: list[float] = []
    alignment: list[float] = []
    penalty: list[float] = []
    count_covered: list[float] = []

    for row_idx, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"entropy_table[{row_idx}]")
        left = _require_number(row, "bin_left", row_idx)
        right = _require_number(row, "bin_right", row_idx)
        if right <= left:
            raise ValueError(
                f"Row {row_idx} has invalid entropy bin bounds: bin_right <= bin_left."
            )
        covered = _require_number(row, "count_covered", row_idx)
        lefts.append(left)
        rights.append(right)
        centers.append((left + right) / 2.0)
        alignment.append(_optional_number(row, "mean_alignment", row_idx))
        penalty.append(_optional_number(row, "mean_penalty", row_idx))
        count_covered.append(covered)

    left_arr = np.asarray(lefts, dtype=np.float64)
    right_arr = np.asarray(rights, dtype=np.float64)
    if np.any(left_arr[1:] < left_arr[:-1]) or np.any(right_arr[1:] < right_arr[:-1]):
        raise ValueError("Entropy bins must appear in nondecreasing order.")

    alignment_arr = np.asarray(alignment, dtype=np.float64)
    penalty_arr = np.asarray(penalty, dtype=np.float64)
    finite_any = np.isfinite(alignment_arr) | np.isfinite(penalty_arr)
    if not np.any(finite_any):
        raise ValueError("No finite mean_alignment or mean_penalty values were found.")

    covered_arr = np.asarray(count_covered, dtype=np.float64)
    total_covered = float(np.sum(covered_arr))
    if total_covered <= 0.0:
        raise ValueError("Covered-token total is zero; cannot render the support strip.")

    return {
        "lefts": left_arr,
        "rights": right_arr,
        "centers": np.asarray(centers, dtype=np.float64),
        "alignment": alignment_arr,
        "penalty": penalty_arr,
        "covered_share": covered_arr / total_covered,
    }


def _plot_curves(output_path: Path, plot_payload: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
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
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    x = plot_payload["centers"]
    lefts = plot_payload["lefts"]
    rights = plot_payload["rights"]
    alignment = plot_payload["alignment"]
    penalty = plot_payload["penalty"]
    covered_share = plot_payload["covered_share"]

    finite_values = np.concatenate(
        [alignment[np.isfinite(alignment)], penalty[np.isfinite(penalty)]]
    )
    curve_max = float(np.max(finite_values)) if finite_values.size else 1.0
    curve_max = max(curve_max, 1e-6)
    strip_height = curve_max * 0.09
    y_top = curve_max * 1.12
    y_bottom = -strip_height * 1.32
    strip_base = -strip_height
    share_max = float(np.max(covered_share))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.25), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.992, bottom=0.29, top=0.82)

    ct._apply_paper_axes_style(ax)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#98A2B3")
        ax.spines[spine_name].set_linewidth(0.9)

    ax.axhline(0.0, color="#B8C1CC", linewidth=0.8, zorder=1)

    for left, right, share in zip(lefts, rights, covered_share):
        alpha = 0.18 if share_max <= 0.0 else 0.18 + 0.42 * (share / share_max)
        rect = Rectangle(
            (left, strip_base),
            right - left,
            strip_height,
            facecolor="#B8C0CA",
            edgecolor="#FFFFFF",
            linewidth=0.5,
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(rect)

    valid_alignment = np.isfinite(alignment)
    valid_penalty = np.isfinite(penalty)
    blue = "#4E79A7"
    red = "#E15759"

    ax.fill_between(
        x,
        0.0,
        alignment,
        where=valid_alignment,
        interpolate=True,
        color=blue,
        alpha=0.18,
        zorder=2,
    )
    ax.fill_between(
        x,
        0.0,
        penalty,
        where=valid_penalty,
        interpolate=True,
        color=red,
        alpha=0.16,
        zorder=2,
    )
    (align_line,) = ax.plot(
        x,
        alignment,
        color=blue,
        linewidth=2.0,
        alpha=0.98,
        zorder=4,
        marker="o",
        markersize=3.2,
        markerfacecolor=blue,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.6,
    )
    (penalty_line,) = ax.plot(
        x,
        penalty,
        color=red,
        linewidth=1.95,
        alpha=0.98,
        zorder=4,
        marker="o",
        markersize=3.2,
        markerfacecolor=red,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.6,
    )

    ax.set_xlim(float(np.min(lefts)), float(np.max(rights)))
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("normalized entropy")
    ax.set_ylabel("mean gain / penalty")
    ax.grid(axis="y", color="#E7ECF2", linewidth=0.55, alpha=0.9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    unique_edges = sorted({float(value) for value in np.concatenate([lefts, rights])})
    if len(unique_edges) <= 9:
        ax.set_xticks(unique_edges)

    ax.text(
        0.0,
        -0.22,
        "gray strip: covered share",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#7A8797",
    )

    legend = fig.legend(
        [align_line, penalty_line],
        ["Alignment gain", "Curvature penalty"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#D7DCE3",
        facecolor="#FFFFFF",
        fontsize=8.7,
        handlelength=2.2,
        handletextpad=0.6,
        borderpad=0.35,
        columnspacing=1.2,
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
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input metrics summary not found: {input_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    rows = _load_entropy_table(input_path)
    plot_payload = _prepare_plot_payload(rows)
    _plot_curves(output_path, plot_payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
