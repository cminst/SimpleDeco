"""Generate paper-style pass@k / maj@k curves for one or more datasets."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _REPO_ROOT / "script"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import compare_trajs as ct


DEFAULT_OUTPUT = _REPO_ROOT / "figure" / "passmaj_curves.pdf"
DEFAULT_DATASETS = ["aime24"]
DEFAULT_METHOD_PATTERNS = {
    "Base": [
        "ckpt/{dataset}/base/*.jsonl",
        "ckpt/{dataset}/base-*/*.jsonl",
    ],
    "MeanShift": [
        "ckpt/{dataset}/meanshift/*.jsonl",
        "ckpt/{dataset}/meanshift-*/*.jsonl",
    ],
    "AutoDeco": [
        "ckpt/{dataset}/autodeco/*.jsonl",
        "ckpt/{dataset}/simpledeco/*.jsonl",
        "ckpt/{dataset}/autodeco-*/*.jsonl",
        "ckpt/{dataset}/simpledeco-*/*.jsonl",
    ],
    "Greedy": [
        "ckpt/{dataset}/greedy/*.jsonl",
        "ckpt/{dataset}/greedy-*/*.jsonl",
    ],
}
METHOD_ORDER = ("Base", "MeanShift", "AutoDeco", "Greedy")


@dataclass(frozen=True)
class MethodSpec:
    label: str
    patterns: list[str]
    greedy: bool = False
    required: bool = False


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    display_name: str
    labels: list[str]
    plot_data: dict[str, list[list[tuple[int, float, float | None]]]]


def _parse_patterns(values: list[str] | None, fallback: list[str]) -> tuple[list[str], bool]:
    parsed = ct._parse_csv_args(values)
    return (parsed if parsed else list(fallback), bool(parsed))


def _parse_datasets(values: list[str] | None, default: list[str]) -> list[str]:
    parsed = ct._parse_csv_args(values)
    datasets = parsed if parsed else list(default)
    cleaned: list[str] = []
    for dataset in datasets:
        token = dataset.strip().strip("/")
        if token:
            cleaned.append(token)
    if not cleaned:
        raise ValueError("At least one dataset must be provided.")
    return cleaned


def _display_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().strip("/").replace("-", "_").upper()
    dataset_display = {
        "GPQA_DIAMOND": "GPQA-Diamond",
        "MMLU_PRO_LITE": "MMLU-Pro-Lite",
    }
    if normalized in dataset_display:
        return dataset_display[normalized]
    return dataset if any(ch.isupper() for ch in dataset) else dataset.upper()


def _render_dataset_patterns(patterns: list[str], dataset: str) -> list[str]:
    return [pattern.replace("{dataset}", dataset) for pattern in patterns]


def _resolve_candidate_paths(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        if not os.path.isabs(expanded):
            expanded = str(_REPO_ROOT / expanded)
        if glob.has_magic(expanded):
            resolved.extend(Path(match) for match in sorted(glob.glob(expanded)))
            continue
        path = Path(expanded)
        if path.exists():
            resolved.append(path)
    return sorted(dict.fromkeys(path.resolve() for path in resolved))


def _resolve_paths_for_spec(spec: MethodSpec, dataset: str, progress: bool) -> list[Path]:
    paths = _resolve_candidate_paths(spec.patterns)
    if paths:
        return paths
    tried = "\n".join(f"  - {pattern}" for pattern in spec.patterns)
    if spec.required:
        raise FileNotFoundError(
            f"No JSONL inputs found for {spec.label} on dataset '{dataset}'. Tried:\n{tried}"
        )
    ct._log(progress, f"Skipping {dataset}/{spec.label}: no inputs matched.")
    return []


def _build_specs(args: argparse.Namespace, dataset: str) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    for label in METHOD_ORDER:
        raw_patterns, required = _parse_patterns(
            getattr(args, label.lower()),
            DEFAULT_METHOD_PATTERNS[label],
        )
        specs.append(
            MethodSpec(
                label=label,
                patterns=_render_dataset_patterns(raw_patterns, dataset),
                greedy=(label == "Greedy"),
                required=required,
            )
        )
    return specs


def _validate_override_patterns(args: argparse.Namespace, datasets: list[str]) -> None:
    if len(datasets) <= 1:
        return
    for label in METHOD_ORDER:
        values = getattr(args, label.lower())
        if not values:
            continue
        patterns = ct._parse_csv_args(values)
        if any("{dataset}" not in pattern for pattern in patterns):
            raise ValueError(
                f"--{label.lower()} overrides must contain '{{dataset}}' when using multiple datasets."
            )


def _subsample_paths(
    specs_with_paths: list[tuple[MethodSpec, list[Path]]],
    seed: int | None,
    progress: bool,
) -> list[tuple[MethodSpec, list[Path]]]:
    non_greedy_counts = [len(paths) for spec, paths in specs_with_paths if not spec.greedy]
    if not non_greedy_counts:
        return specs_with_paths
    target = min(non_greedy_counts)
    if len(set(non_greedy_counts)) == 1:
        return specs_with_paths

    ct._log(progress, f"Subsampling non-greedy groups to {target} seed file(s).")
    trimmed: list[tuple[MethodSpec, list[Path]]] = []
    for spec, paths in specs_with_paths:
        if spec.greedy or len(paths) <= target:
            trimmed.append((spec, paths))
            continue
        sampled = ct._sample_paths(paths, target, seed)
        ct._log(progress, f"  {spec.label}: {len(paths)} -> {len(sampled)} files")
        trimmed.append((spec, sampled))
    return trimmed


def _load_groups(
    dataset: str,
    specs: list[MethodSpec],
    seed: int | None,
    greedy_samples: int,
    progress: bool,
) -> tuple[
    list[tuple[str, list[dict[str, list[float]]], dict[str, list[float]]]],
    list[str],
    list[str],
    int,
]:
    specs_with_paths = []
    for spec in specs:
        paths = _resolve_paths_for_spec(spec, dataset, progress)
        if paths:
            specs_with_paths.append((spec, paths))

    if not specs_with_paths:
        raise FileNotFoundError(f"No JSONL inputs found for dataset '{dataset}'.")

    specs_with_paths = _subsample_paths(specs_with_paths, seed, progress)

    counts = [len(paths) for _, paths in specs_with_paths]
    target_seeds = min(counts) if counts else 0
    groups: list[tuple[str, list[dict[str, list[float]]], dict[str, list[float]]]] = []

    for spec, paths in specs_with_paths:
        ct._log(progress, f"Loading {dataset}/{spec.label}: {len(paths)} file(s)")
        scores_list: list[dict[str, list[float]]] = []
        for idx, path in enumerate(paths, 1):
            ct._log(progress, f"  [{idx}/{len(paths)}] {path}")
            scores_list.append(ct._load_scores(path, progress=progress))

        pooled = ct._merge_scores(paths)
        if not pooled:
            raise RuntimeError(f"Failed to load scores for {dataset}/{spec.label}.")

        if spec.greedy:
            scores_list = [ct._expand_samples(scores, greedy_samples) for scores in scores_list]
            pooled = ct._expand_samples(pooled, greedy_samples)
            if target_seeds and len(scores_list) != target_seeds:
                if len(scores_list) < target_seeds:
                    scores_list = [
                        scores_list[idx % len(scores_list)] for idx in range(target_seeds)
                    ]
                else:
                    scores_list = scores_list[:target_seeds]

        groups.append((spec.label, scores_list, pooled))

    all_dicts: list[dict[str, list[float]]] = []
    for _, scores_list, _ in groups:
        all_dicts.extend(scores_list)
    if not all_dicts:
        raise RuntimeError(f"No scores loaded from dataset '{dataset}'.")

    common = set(all_dicts[0])
    for scores in all_dicts[1:]:
        common &= set(scores)
    if not common:
        raise RuntimeError(f"No overlapping problems across methods for dataset '{dataset}'.")

    min_samples_pooled = min(len(group[2][pid]) for group in groups for pid in common)
    if min_samples_pooled < 1:
        raise RuntimeError(f"Insufficient samples to compute metrics for dataset '{dataset}'.")

    labels = [label for label, _, _ in groups]
    return groups, labels, sorted(common), min_samples_pooled


def _method_style(label: str, fallback_color: str) -> dict[str, Any]:
    lower = label.lower()
    if "greedy" in lower:
        return {
            "color": "#98A1AB",
            "alpha": 0.9,
            "linewidth": 1.4,
            "linestyle": (0, (4.2, 2.4)),
            "marker": None,
            "markersize": 0.0,
            "markeredgewidth": 0.0,
            "zorder": 2,
        }
    if "meanshift" in lower:
        color = "#59A14F"
    elif "autodeco" in lower or "simpledeco" in lower:
        color = "#B07AA1"
    elif "base" in lower:
        color = "#4E79A7"
    else:
        color = fallback_color
    return {
        "color": color,
        "alpha": 0.97,
        "linewidth": 1.9,
        "linestyle": "-",
        "marker": None,
        "markersize": 0.0,
        "markeredgewidth": 0.0,
        "zorder": 3,
    }


def _prepare_dataset_results(
    args: argparse.Namespace,
    datasets: list[str],
) -> tuple[list[DatasetResult], list[str]]:
    dataset_results: list[DatasetResult] = []
    label_order: list[str] = []

    for dataset in datasets:
        specs = _build_specs(args, dataset)
        groups, labels, common, min_samples_pooled = _load_groups(
            dataset,
            specs,
            seed=args.seed,
            greedy_samples=args.greedy_samples,
            progress=args.progress,
        )
        effective_max_k = min(args.max_k, min_samples_pooled)
        configs = [("maj", k) for k in range(1, effective_max_k + 1)]
        configs += [("pass", k) for k in range(1, effective_max_k + 1)]

        ct._log(args.progress, f"Computing metrics for {dataset}: {len(configs)} configs")
        _, plot_data = ct._compute_metrics(groups, configs, common)
        dataset_results.append(
            DatasetResult(
                dataset=dataset,
                display_name=_display_dataset_name(dataset),
                labels=labels,
                plot_data=plot_data,
            )
        )
        for label in labels:
            if label not in label_order:
                label_order.append(label)

    return dataset_results, label_order


def _add_row_titles(fig: Any, axes: Any, results: list[DatasetResult]) -> None:
    from matplotlib.lines import Line2D

    for row_idx, result in enumerate(results):
        left_pos = axes[row_idx, 0].get_position()
        right_pos = axes[row_idx, 1].get_position()
        center_x = (left_pos.x0 + right_pos.x1) / 2.0
        title_y = left_pos.y1 + 0.018
        fig.text(
            center_x,
            title_y,
            result.display_name,
            ha="center",
            va="bottom",
            fontsize=10.4,
            fontweight="bold",
            color="#344054",
        )
        if row_idx > 0:
            prev_bottom = axes[row_idx - 1, 0].get_position().y0
            current_top = left_pos.y1
            separator_y = current_top + (prev_bottom - current_top) * 0.48
            fig.add_artist(
                Line2D(
                    [left_pos.x0, right_pos.x1],
                    [separator_y, separator_y],
                    transform=fig.transFigure,
                    color="#E7ECF2",
                    linewidth=0.8,
                    zorder=0.1,
                )
            )


def _plot_passmaj_curves(
    output_path: Path,
    dataset_results: list[DatasetResult],
    legend_labels: list[str],
    maj_avg: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to generate the pass@k / maj@k plot.") from exc

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{mathpazo}",
            "font.family": "serif",
            "font.serif": ["Palatino", "Times New Roman", "Times"],
            "font.size": 9.4,
            "axes.labelsize": 9.8,
            "axes.titlesize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    prepared_results: list[dict[str, list[list[tuple[int, float, float | None]]]]] = []
    global_ks = {"maj": set(), "pass": set()}
    for result in dataset_results:
        prepared = {
            mode: [ct._prepare_plot_series(mode, series, maj_avg) for series in result.plot_data.get(mode, [])]
            for mode in ("maj", "pass")
        }
        prepared_results.append(prepared)
        for mode in ("maj", "pass"):
            for series in prepared[mode]:
                for k, _, _ in series:
                    global_ks[mode].add(k)

    all_ks = sorted(global_ks["maj"] | global_ks["pass"])
    x_min = min(all_ks) if all_ks else 1
    x_max = max(all_ks) if all_ks else 1
    x_span = x_max - x_min
    left_pad = min(0.4, max(0.18, (x_span if x_span > 0 else 1.0) * 0.05))
    right_pad = min(0.4, max(0.18, (x_span if x_span > 0 else 1.0) * 0.05))
    palette = ct._paper_palette(max(1, len(legend_labels)))
    maj_ticks = sorted(global_ks["maj"]) or [1]
    pass_ticks = sorted(global_ks["pass"]) or [1]

    nrows = len(dataset_results)
    fig_height = 0.7 + nrows * 2.0
    fig, axes = plt.subplots(nrows, 2, figsize=(5.7, fig_height), squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.992, bottom=0.075, top=0.86, wspace=0.22, hspace=0.68)

    legend_handles: dict[str, Any] = {}
    mode_meta = [("maj", r"maj@k (\%)"), ("pass", r"pass@k (\%)")]
    for row_idx, result in enumerate(dataset_results):
        prepared = prepared_results[row_idx]
        for col_idx, (mode, title) in enumerate(mode_meta):
            ax = axes[row_idx, col_idx]
            ct._apply_paper_axes_style(ax)
            for spine_name in ("top", "right", "left", "bottom"):
                ax.spines[spine_name].set_visible(True)
                ax.spines[spine_name].set_color("#98A2B3")
                ax.spines[spine_name].set_linewidth(0.9)

            series_groups = prepared.get(mode, [])
            mode_ks: set[int] = set()
            for idx, series in enumerate(series_groups):
                if not series:
                    continue
                ks = [k for k, _, _ in series]
                vals = [value for _, value, _ in series]
                mode_ks.update(ks)
                label = result.labels[idx]
                style = _method_style(label, palette[idx % len(palette)])
                (line,) = ax.plot(
                    ks,
                    vals,
                    color=style["color"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    markeredgewidth=style["markeredgewidth"],
                    markeredgecolor="#FFFFFF",
                    markerfacecolor=style["color"],
                )
                legend_handles.setdefault(label, line)

            if mode_ks:
                ax.grid(axis="y", color="#E7ECF2", linewidth=0.55, alpha=0.9)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.set_xlim(max(0.0, x_min - left_pad), x_max + right_pad)
                ax.set_xticks(maj_ticks if mode == "pass" and maj_ticks else (maj_ticks if mode == "maj" else pass_ticks))
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()

            if row_idx == 0:
                ax.set_title(title, pad=8.5)
            if row_idx == nrows - 1:
                ax.set_xlabel("sample budget $k$")

    if not legend_handles:
        raise RuntimeError("No plot data available for any dataset.")

    _add_row_titles(fig, axes, dataset_results)

    legend = fig.legend(
        [legend_handles[label] for label in legend_labels if label in legend_handles],
        [label for label in legend_labels if label in legend_handles],
        loc="upper center",
        ncol=min(4, len(legend_handles)),
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


def build_arg_parser(
    default_datasets: list[str] | None = None,
    default_output: Path | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate pass@k / maj@k paper curves for one or more datasets."
    )
    parser.add_argument(
        "--datasets",
        action="append",
        help="Dataset name(s), comma-separated if needed (default: aime24).",
    )
    parser.add_argument(
        "--base",
        action="append",
        help=(
            "Comma-separated JSONL path(s)/glob(s) for Base. "
            "Use {dataset} in the pattern when combining multiple datasets."
        ),
    )
    parser.add_argument(
        "--meanshift",
        action="append",
        help=(
            "Comma-separated JSONL path(s)/glob(s) for MeanShift. "
            "Use {dataset} in the pattern when combining multiple datasets."
        ),
    )
    parser.add_argument(
        "--autodeco",
        action="append",
        help=(
            "Comma-separated JSONL path(s)/glob(s) for AutoDeco. "
            "Use {dataset} in the pattern when combining multiple datasets."
        ),
    )
    parser.add_argument(
        "--greedy",
        action="append",
        help=(
            "Comma-separated JSONL path(s)/glob(s) for Greedy. "
            "Use {dataset} in the pattern when combining multiple datasets."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(default_output or DEFAULT_OUTPUT),
        help=f"Output PDF/PNG path (default: {default_output or DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used if non-greedy methods need seed-count subsampling.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=16,
        help="Maximum k to include in the curves (default: 16).",
    )
    parser.add_argument(
        "--maj-avg",
        choices=("pairs", "odd", "all"),
        default="pairs",
        help="maj@k display mode (default: pairs).",
    )
    parser.add_argument(
        "--greedy-samples",
        type=int,
        default=16,
        help="Synthetic sample count assigned to greedy runs (default: 16).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print resolved inputs and loading progress.",
    )
    parser.set_defaults(_default_datasets=default_datasets or DEFAULT_DATASETS)
    return parser


def main(
    argv: list[str] | None = None,
    default_datasets: list[str] | None = None,
    default_output: Path | None = None,
) -> None:
    parser = build_arg_parser(default_datasets=default_datasets, default_output=default_output)
    args = parser.parse_args(argv)

    if args.max_k < 1:
        raise ValueError("--max-k must be >= 1.")
    if args.greedy_samples < 1:
        raise ValueError("--greedy-samples must be >= 1.")

    datasets = _parse_datasets(args.datasets, args._default_datasets)
    _validate_override_patterns(args, datasets)

    dataset_results, legend_labels = _prepare_dataset_results(args, datasets)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    _plot_passmaj_curves(output_path, dataset_results, legend_labels, args.maj_avg)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
