"""Plot paper-style histograms of predicted temperature and top-p values."""
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


DEFAULT_OUTPUT = _REPO_ROOT / "figure" / "pertoken_temp_top_p_histogram.pdf"
DEFAULT_DATASET_SPLIT = "tokens"
DEFAULT_TEMP_BINS = 24
DEFAULT_TOP_P_BINS = 20


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
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"Column '{column_name}' does not contain any finite values.")
    return finite_values


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


def _style_axes(ax: Any) -> None:
    ct._apply_paper_axes_style(ax)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#98A2B3")
        ax.spines[spine_name].set_linewidth(0.9)


def _compute_histogram(
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full(values.shape, 100.0 / float(values.size), dtype=np.float64)
    hist, resolved_edges = np.histogram(values, bins=edges, weights=weights)
    widths = np.diff(resolved_edges)
    return hist, widths


def _plot_histograms(
    output_path: Path,
    temperatures: np.ndarray,
    top_ps: np.ndarray,
    temp_bins: int,
    top_p_bins: int,
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

    temp_edges = np.linspace(0.0, 2.0, temp_bins + 1, dtype=np.float64)
    top_p_edges = np.linspace(0.0, 1.0, top_p_bins + 1, dtype=np.float64)
    temp_hist, temp_widths = _compute_histogram(temperatures, temp_edges)
    top_p_hist, top_p_widths = _compute_histogram(top_ps, top_p_edges)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.8),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.0, 1.0)},
    )
    fig.subplots_adjust(left=0.09, right=0.992, bottom=0.345, top=0.915, wspace=0.24)

    _style_axes(ax_left)
    _style_axes(ax_right)

    blue = "#4E79A7"
    plum = "#B07AA1"

    ax_left.bar(
        temp_edges[:-1],
        temp_hist,
        width=temp_widths,
        align="edge",
        color=blue,
        edgecolor="#FFFFFF",
        linewidth=0.65,
        alpha=0.95,
        zorder=3,
    )
    ax_right.bar(
        top_p_edges[:-1],
        top_p_hist,
        width=top_p_widths,
        align="edge",
        color=plum,
        edgecolor="#FFFFFF",
        linewidth=0.65,
        alpha=0.92,
        zorder=3,
    )

    for axis in (ax_left, ax_right):
        axis.grid(axis="y", color="#E7ECF2", linewidth=0.55, alpha=0.9)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis="x", pad=1.0)

    ax_left.set_xlim(0.0, 2.0)
    ax_right.set_xlim(0.0, 1.0)

    ax_left.set_xticks(np.linspace(0.0, 2.0, 5))
    ax_right.set_xticks(np.linspace(0.0, 1.0, 6))

    ymax = max(
        float(np.max(temp_hist)) if temp_hist.size else 0.0,
        float(np.max(top_p_hist)) if top_p_hist.size else 0.0,
    )
    upper = ymax * 1.12 if ymax > 0.0 else 1.0
    ax_left.set_ylim(0.0, upper)
    ax_right.set_ylim(0.0, upper)

    ax_left.set_ylabel(r"share of tokens (\%)")
    ax_left.set_xlabel("temperature", labelpad=4)
    ax_right.set_xlabel(r"top-$p$", labelpad=4)

    ax_left.text(
        0.5,
        -0.34,
        r"(a) Predicted temperature",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#475467",
        clip_on=False,
    )
    ax_right.text(
        0.5,
        -0.34,
        r"(b) Predicted top-$p$",
        transform=ax_right.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#475467",
        clip_on=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style histograms for per-token predicted temperature and top-p.",
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
        "--temp-bins",
        type=int,
        default=DEFAULT_TEMP_BINS,
        help="Number of histogram bins for T_hat across [0, 2].",
    )
    parser.add_argument(
        "--top-p-bins",
        type=int,
        default=DEFAULT_TOP_P_BINS,
        help="Number of histogram bins for p_hat across [0, 1].",
    )
    args = parser.parse_args()

    if args.temp_bins <= 0:
        raise ValueError("--temp-bins must be positive.")
    if args.top_p_bins <= 0:
        raise ValueError("--top-p-bins must be positive.")

    input_path = _resolve_input_path(args.input)
    output_path = _resolve_input_path(args.output)

    tokens_dataset = _load_tokens_dataset(input_path, split_name=args.split)
    temperatures = _validate_bounded_values(
        _load_numeric_column(tokens_dataset, "T_hat"),
        name="T_hat",
        lower=0.0,
        upper=2.0,
    )
    top_ps = _validate_bounded_values(
        _load_numeric_column(tokens_dataset, "p_hat"),
        name="p_hat",
        lower=0.0,
        upper=1.0,
    )

    _plot_histograms(
        output_path=output_path,
        temperatures=temperatures,
        top_ps=top_ps,
        temp_bins=int(args.temp_bins),
        top_p_bins=int(args.top_p_bins),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
