"""Generate the paper-specific AIME24 pass@k / maj@k curves."""
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


DEFAULT_OUTPUT = _REPO_ROOT / "figure" / "aime24_passmaj_curves.pdf"
DEFAULT_PATTERN_CANDIDATES = {
    "Base": [
        "ckpt/aime24/base/*.jsonl",
        "ckpt/aime24/base-r1-distill-qwen7b/*.jsonl",
    ],
    "MeanShift": [
        "ckpt/aime24/meanshift/*.jsonl",
        "ckpt/aime24/meanshift2-r1-distill-qwen7b/*.jsonl",
    ],
    "AutoDeco": [
        "ckpt/aime24/autodeco/*.jsonl",
        "ckpt/aime24/simpledeco/*.jsonl",
        "ckpt/aime24/autodeco-r1-distill-qwen7b/*.jsonl",
    ],
    "Greedy": [
        "ckpt/aime24/greedy/*.jsonl",
        "ckpt/aime24/greedy-r1-distill-qwen7b/*.jsonl",
    ],
}


@dataclass(frozen=True)
class MethodSpec:
    label: str
    patterns: list[str]
    greedy: bool = False


def _parse_patterns(values: list[str] | None, fallback: list[str]) -> list[str]:
    parsed = ct._parse_csv_args(values)
    return parsed if parsed else list(fallback)


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
    unique = sorted(dict.fromkeys(path.resolve() for path in resolved))
    return unique


def _resolve_required_paths(label: str, patterns: list[str]) -> list[Path]:
    paths = _resolve_candidate_paths(patterns)
    if paths:
        return paths
    tried = "\n".join(f"  - {pattern}" for pattern in patterns)
    raise FileNotFoundError(
        f"No JSONL inputs found for {label}. Tried:\n{tried}\n"
        "Pass explicit --base/--meanshift/--autodeco/--greedy globs if your files live elsewhere."
    )


def _build_specs(args: argparse.Namespace) -> list[MethodSpec]:
    return [
        MethodSpec(
            label="Base",
            patterns=_parse_patterns(args.base, DEFAULT_PATTERN_CANDIDATES["Base"]),
        ),
        MethodSpec(
            label="MeanShift",
            patterns=_parse_patterns(args.meanshift, DEFAULT_PATTERN_CANDIDATES["MeanShift"]),
        ),
        MethodSpec(
            label="AutoDeco",
            patterns=_parse_patterns(args.autodeco, DEFAULT_PATTERN_CANDIDATES["AutoDeco"]),
        ),
        MethodSpec(
            label="Greedy",
            patterns=_parse_patterns(args.greedy, DEFAULT_PATTERN_CANDIDATES["Greedy"]),
            greedy=True,
        ),
    ]


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
    specs_with_paths = [
        (spec, _resolve_required_paths(spec.label, spec.patterns))
        for spec in specs
    ]
    specs_with_paths = _subsample_paths(specs_with_paths, seed, progress)

    counts = [len(paths) for _, paths in specs_with_paths]
    target_seeds = min(counts) if counts else 0
    groups: list[tuple[str, list[dict[str, list[float]]], dict[str, list[float]]]] = []

    for spec, paths in specs_with_paths:
        ct._log(progress, f"Loading {spec.label}: {len(paths)} file(s)")
        scores_list: list[dict[str, list[float]]] = []
        for idx, path in enumerate(paths, 1):
            ct._log(progress, f"  [{idx}/{len(paths)}] {path}")
            scores_list.append(ct._load_scores(path, progress=progress))

        pooled = ct._merge_scores(paths)
        if not pooled:
            raise RuntimeError(f"Failed to load scores for {spec.label}.")

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
        raise RuntimeError("No scores loaded from inputs.")

    common = set(all_dicts[0])
    for scores in all_dicts[1:]:
        common &= set(scores)
    if not common:
        raise RuntimeError("No overlapping problems across the selected methods.")

    min_samples_pooled = min(
        min(len(group[2][pid]) for group in groups)
        for pid in common
    )
    if min_samples_pooled < 1:
        raise RuntimeError("Insufficient samples to compute metrics.")

    labels = [label for label, _, _ in groups]
    return groups, labels, sorted(common), min_samples_pooled


def _method_style(label: str, fallback_color: str) -> dict[str, Any]:
    lower = label.lower()
    if "greedy" in lower:
        return {
            "color": "#98A1AB",
            "text_color": "#6B7280",
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
        "text_color": color,
        "alpha": 0.97,
        "linewidth": 2.2,
        "linestyle": "-",
        "marker": None,
        "markersize": 0.0,
        "markeredgewidth": 0.0,
        "zorder": 3,
    }


def _draw_line_end_labels_for_figure(ax: Any, entries: list[dict[str, Any]], pe: Any) -> None:
    if not entries:
        return

    x_values = [entry["x"] for entry in entries]
    x_min = min(x_values)
    x_max = max(x_values)
    span = x_max - x_min
    pad = max(0.5, (span if span > 0 else 1.0) * 0.1)
    label_x = x_max + pad * 0.3
    connector_x = label_x - pad * 0.1

    current_left, current_right = ax.get_xlim()
    left_limit = min(current_left, x_min)
    ax.set_xlim(left_limit, max(current_right, x_max + pad * 1.02))

    y_positions = ct._resolve_label_positions(
        [(idx, entry["y"]) for idx, entry in enumerate(entries)],
        *ax.get_ylim(),
    )
    texts: list[Any] = []
    for idx, entry in enumerate(entries):
        label_y = y_positions[idx]
        ax.plot(
            [entry["x"], connector_x],
            [entry["y"], label_y],
            color=entry["color"],
            alpha=min(0.7, entry["alpha"] * 0.7),
            linewidth=0.8,
            zorder=entry["zorder"],
            solid_capstyle="round",
        )
        text = ax.text(
            label_x,
            label_y,
            entry["label"],
            ha="left",
            va="center",
            fontsize=8.9,
            color=entry["text_color"],
            alpha=entry["alpha"],
            weight=entry["weight"],
            zorder=entry["zorder"] + 0.1,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white", alpha=0.92)])
        texts.append(text)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    max_text_width = max(text.get_window_extent(renderer=renderer).width for text in texts)
    target_right_padding_px = 8.0
    denominator = axes_bbox.width - max_text_width - target_right_padding_px
    if denominator > 1.0:
        desired_right = left_limit + ((label_x - left_limit) * axes_bbox.width / denominator)
        min_right = x_max + pad * 0.68
        ax.set_xlim(left_limit, max(min_right, desired_right))


def _plot_aime24_curves(
    output_path: Path,
    plot_data: dict[str, list[list[tuple[int, float, float | None]]]],
    labels: list[str],
    maj_avg: str,
) -> None:
    try:
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to generate the AIME24 plot.") from exc

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

    prepared_plot_data = {
        mode: [ct._prepare_plot_series(mode, series, maj_avg) for series in plot_data.get(mode, [])]
        for mode in ("maj", "pass")
    }

    all_ks = [
        k
        for series_groups in prepared_plot_data.values()
        for series in series_groups
        for k, _, _ in series
    ]
    x_min = min(all_ks) if all_ks else 1
    x_max = max(all_ks) if all_ks else 1
    x_span = x_max - x_min
    left_pad = min(0.4, max(0.18, (x_span if x_span > 0 else 1.0) * 0.05))
    label_pad = max(0.8, (x_span if x_span > 0 else 1.0) * 0.14)
    palette = ct._paper_palette(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), constrained_layout=True)
    for ax, (mode, ylabel) in zip(axes, [("maj", r"maj@k (\%)"), ("pass", r"pass@k (\%)")]):
        ct._apply_paper_axes_style(ax)
        series_groups = prepared_plot_data.get(mode, [])
        label_entries: list[dict[str, Any]] = []
        mode_ks: set[int] = set()

        for idx, series in enumerate(series_groups):
            if not series:
                continue
            ks = [k for k, _, _ in series]
            vals = [value for _, value, _ in series]
            mode_ks.update(ks)
            style = _method_style(labels[idx], palette[idx % len(palette)])
            ax.plot(
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
            label_entries.append(
                {
                    "label": labels[idx],
                    "x": ks[-1],
                    "y": vals[-1],
                    "color": style["color"],
                    "text_color": style["text_color"],
                    "alpha": style["alpha"],
                    "weight": "normal",
                    "zorder": style["zorder"],
                }
            )

        if not label_entries:
            raise RuntimeError(f"No plot data available for {mode}@k.")

        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D7DCE3", linewidth=0.75, alpha=0.55)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlim(max(0.0, x_min - left_pad), x_max + label_pad * 1.6)
        ax.set_xticks(sorted(mode_ks))
        _draw_line_end_labels_for_figure(ax, label_entries, pe)

    for ax in axes:
        ax.set_xlabel("sample budget $k$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the AIME24 pass@k / maj@k paper figure."
    )
    parser.add_argument(
        "--base",
        action="append",
        help="Comma-separated JSONL path(s)/glob(s) for Base seeds.",
    )
    parser.add_argument(
        "--meanshift",
        action="append",
        help="Comma-separated JSONL path(s)/glob(s) for MeanShift seeds.",
    )
    parser.add_argument(
        "--autodeco",
        action="append",
        help="Comma-separated JSONL path(s)/glob(s) for AutoDeco seeds.",
    )
    parser.add_argument(
        "--greedy",
        action="append",
        help="Comma-separated JSONL path(s)/glob(s) for Greedy runs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PDF/PNG path (default: {DEFAULT_OUTPUT}).",
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
    args = parser.parse_args()

    if args.max_k < 1:
        raise ValueError("--max-k must be >= 1.")
    if args.greedy_samples < 1:
        raise ValueError("--greedy-samples must be >= 1.")

    specs = _build_specs(args)
    groups, labels, common, min_samples_pooled = _load_groups(
        specs,
        seed=args.seed,
        greedy_samples=args.greedy_samples,
        progress=args.progress,
    )

    effective_max_k = min(args.max_k, min_samples_pooled)
    configs = [("maj", k) for k in range(1, effective_max_k + 1)]
    configs += [("pass", k) for k in range(1, effective_max_k + 1)]

    ct._log(args.progress, f"Computing metrics for {len(configs)} configs")
    _, plot_data = ct._compute_metrics(groups, configs, common)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    _plot_aime24_curves(output_path, plot_data, labels, args.maj_avg)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
