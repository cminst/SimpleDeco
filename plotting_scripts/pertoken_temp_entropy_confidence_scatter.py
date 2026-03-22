"""Plot paper-style trend figures for AutoDeco temperature vs entropy/confidence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datasets import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _REPO_ROOT / "script"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import compare_trajs as ct


DEFAULT_OUTPUT = _REPO_ROOT / "figure" / "pertoken_temp_entropy_confidence_trends.pdf"
DEFAULT_DATASET_SPLIT = "tokens"
DEFAULT_MAX_POINTS = 0
DEFAULT_SEED = 0
DEFAULT_NUM_BINS = 28
DEFAULT_MIN_BIN_COUNT = 64


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _load_tokens_dataset(input_path: Path, split_name: str | None) -> "Dataset":
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to load the Hugging Face dataset input.") from exc

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {input_path}")

    dataset = load_from_disk(str(input_path))
    if isinstance(dataset, Dataset):
        return dataset
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a Dataset or DatasetDict at {input_path}, got {type(dataset)!r}.")

    resolved_split = split_name or DEFAULT_DATASET_SPLIT
    if resolved_split in dataset:
        return dataset[resolved_split]
    if DEFAULT_DATASET_SPLIT in dataset:
        return dataset[DEFAULT_DATASET_SPLIT]
    if len(dataset) == 1:
        return next(iter(dataset.values()))

    available = ", ".join(dataset.keys())
    raise KeyError(
        f"Could not resolve a token split from {input_path}. "
        f"Requested '{resolved_split}'. Available splits: {available}"
    )


def _load_numeric_column(dataset: "Dataset", column_name: str) -> np.ndarray:
    if column_name not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise KeyError(f"Column '{column_name}' not found. Available columns: {available}")
    values = np.asarray(dataset[column_name], dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"Column '{column_name}' is empty.")
    return values


def _validate_bounded_values(
    values: np.ndarray,
    *,
    name: str,
    lower: float,
    upper: float,
    atol: float = 1e-6,
) -> np.ndarray:
    if np.any(values < lower - atol) or np.any(values > upper + atol):
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        raise ValueError(
            f"Column '{name}' contains values outside [{lower}, {upper}]. "
            f"Observed range: [{observed_min:.6g}, {observed_max:.6g}]"
        )
    return np.clip(values, lower, upper)


def _validate_nonnegative_values(
    values: np.ndarray,
    *,
    name: str,
    atol: float = 1e-6,
) -> np.ndarray:
    if np.any(values < -atol):
        observed_min = float(np.min(values))
        raise ValueError(f"Column '{name}' contains negative values. Observed min: {observed_min:.6g}")
    return np.clip(values, 0.0, None)


def _filter_finite_rows(
    temperatures: np.ndarray,
    entropies: np.ndarray,
    confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(temperatures) & np.isfinite(entropies) & np.isfinite(confidences)
    if not np.any(finite_mask):
        raise ValueError("No rows remain after dropping non-finite values from T_hat, H, and p_max.")
    return temperatures[finite_mask], entropies[finite_mask], confidences[finite_mask]


def _maybe_subsample_points(
    temperatures: np.ndarray,
    entropies: np.ndarray,
    confidences: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points <= 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty

    total_points = temperatures.size
    if total_points <= max_points:
        return temperatures, entropies, confidences

    rng = np.random.default_rng(seed)
    indices = rng.choice(total_points, size=max_points, replace=False)
    indices.sort()
    return temperatures[indices], entropies[indices], confidences[indices]


def _style_axes(ax: Any) -> None:
    ct._apply_paper_axes_style(ax)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#98A2B3")
        ax.spines[spine_name].set_linewidth(0.9)
    ax.spines["bottom"].set_zorder(6)


def _compute_equal_count_trend(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    num_bins: int,
    min_bin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have the same length.")
    if x_values.size == 0:
        raise ValueError("Cannot compute a trend from zero points.")

    order = np.argsort(x_values, kind="mergesort")
    x_sorted = x_values[order]
    y_sorted = y_values[order]
    effective_bins = min(num_bins, max(1, x_sorted.size // min_bin_count))
    if effective_bins <= 1:
        effective_bins = 1

    x_centers: list[float] = []
    y_medians: list[float] = []
    y_q1: list[float] = []
    y_q3: list[float] = []

    bin_indices = np.array_split(np.arange(x_sorted.size, dtype=np.int64), effective_bins)
    for idx in bin_indices:
        if idx.size < min_bin_count and len(bin_indices) > 1:
            continue
        x_chunk = x_sorted[idx]
        y_chunk = y_sorted[idx]
        x_centers.append(float(np.median(x_chunk)))
        y_medians.append(float(np.median(y_chunk)))
        y_q1.append(float(np.quantile(y_chunk, 0.25)))
        y_q3.append(float(np.quantile(y_chunk, 0.75)))

    if not x_centers:
        raise ValueError(
            "No trend bins remain after applying --min-bin-count. "
            "Reduce --min-bin-count or --num-bins."
        )

    return (
        np.asarray(x_centers, dtype=np.float64),
        np.asarray(y_medians, dtype=np.float64),
        np.asarray(y_q1, dtype=np.float64),
        np.asarray(y_q3, dtype=np.float64),
    )


def _plot_panel(
    ax: Any,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_background: np.ndarray,
    y_background: np.ndarray,
    color: str,
    xlabel: str,
    xlim: tuple[float, float] | None,
    num_bins: int,
    min_bin_count: int,
) -> None:
    x_centers, y_medians, y_q1, y_q3 = _compute_equal_count_trend(
        x_values,
        y_values,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )

    if x_background.size > 0 and y_background.size > 0:
        ax.scatter(
            x_background,
            y_background,
            s=3.0,
            alpha=0.06,
            color="#667085",
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )

    ax.fill_between(
        x_centers,
        y_q1,
        y_q3,
        color=color,
        alpha=0.20,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        x_centers,
        y_medians,
        color=color,
        linewidth=1.9,
        zorder=3,
    )
    ax.scatter(
        x_centers,
        y_medians,
        s=10.0,
        color=color,
        edgecolors="#FFFFFF",
        linewidths=0.35,
        zorder=4,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=6)


def _plot_trends(
    output_path: Path,
    temperatures: np.ndarray,
    entropies: np.ndarray,
    confidences: np.ndarray,
    background_temperatures: np.ndarray,
    background_entropies: np.ndarray,
    background_confidences: np.ndarray,
    *,
    num_bins: int,
    min_bin_count: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
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

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.25),
        constrained_layout=False,
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.0)},
    )
    fig.subplots_adjust(left=0.11, right=0.992, bottom=0.25, top=0.91, wspace=0.16)

    _style_axes(ax_left)
    _style_axes(ax_right)

    trend_color = "#D39A6A"

    for axis in (ax_left, ax_right):
        axis.tick_params(axis="x", pad=1.0)

    _plot_panel(
        ax_left,
        x_values=entropies,
        y_values=temperatures,
        x_background=background_entropies,
        y_background=background_temperatures,
        color=trend_color,
        xlabel=r"(a) Entropy $H$",
        xlim=None,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )
    _plot_panel(
        ax_right,
        x_values=confidences,
        y_values=temperatures,
        x_background=background_confidences,
        y_background=background_temperatures,
        color=trend_color,
        xlabel=r"(b) Top-1 confidence $p_{\max}$",
        xlim=None,
        num_bins=num_bins,
        min_bin_count=min_bin_count,
    )

    ax_left.set_ylabel(r"Predicted temperature")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style trend plots for per-token temperature vs entropy/confidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a save_to_disk Hugging Face Dataset or DatasetDict from collect_pertoken_diagnostics.py.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to use when the input is a DatasetDict.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PDF/PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=(
            "Maximum number of faint background points to draw. "
            "Use 0 to disable the raw-point backdrop and only show the trend."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic background-point subsampling.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Approximate number of equal-count bins used to summarize the temperature trend.",
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=DEFAULT_MIN_BIN_COUNT,
        help="Minimum number of tokens required for a trend bin to be shown.",
    )
    args = parser.parse_args()

    if args.max_points < 0:
        raise ValueError("--max-points must be nonnegative.")
    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive.")
    if args.min_bin_count <= 0:
        raise ValueError("--min-bin-count must be positive.")

    input_path = _resolve_input_path(args.input)
    output_path = _resolve_input_path(args.output)

    tokens_dataset = _load_tokens_dataset(input_path, split_name=args.split)
    temperatures_raw = _load_numeric_column(tokens_dataset, "T_hat")
    entropies_raw = _load_numeric_column(tokens_dataset, "H")
    confidences_raw = _load_numeric_column(tokens_dataset, "p_max")

    temperatures, entropies, confidences = _filter_finite_rows(
        temperatures_raw,
        entropies_raw,
        confidences_raw,
    )
    temperatures = _validate_bounded_values(
        temperatures,
        name="T_hat",
        lower=0.0,
        upper=2.0,
    )
    entropies = _validate_nonnegative_values(entropies, name="H")
    confidences = _validate_bounded_values(
        confidences,
        name="p_max",
        lower=0.0,
        upper=1.0,
    )
    background_temperatures, background_entropies, background_confidences = _maybe_subsample_points(
        temperatures,
        entropies,
        confidences,
        max_points=int(args.max_points),
        seed=int(args.seed),
    )

    _plot_trends(
        output_path=output_path,
        temperatures=temperatures,
        entropies=entropies,
        confidences=confidences,
        background_temperatures=background_temperatures,
        background_entropies=background_entropies,
        background_confidences=background_confidences,
        num_bins=int(args.num_bins),
        min_bin_count=int(args.min_bin_count),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
