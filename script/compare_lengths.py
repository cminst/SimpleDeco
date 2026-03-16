"""Compare response token lengths across JSONL trajectory files."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} in {path}") from exc
            if isinstance(row, dict):
                yield row


def _iter_response_texts(row: Dict[str, Any]) -> Iterable[str]:
    list_keys = ("solutions", "responses", "completions", "outputs", "generated")
    for key in list_keys:
        values = row.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    yield value
            return

    scalar_keys = ("response", "completion", "solution", "output", "generated")
    for key in scalar_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            yield value
            return


def _load_responses(paths: Sequence[Path], progress: bool = False) -> List[str]:
    responses: List[str] = []
    for path_idx, path in enumerate(paths, 1):
        _log(progress, f"[{path_idx}/{len(paths)}] Loading {path}")
        for row in _read_jsonl(path):
            responses.extend(_iter_response_texts(row))
    return responses


def _resolve_inputs(value: str) -> List[Path]:
    expanded = os.path.expanduser(value)
    if glob.has_magic(expanded):
        matches = sorted(glob.glob(expanded))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {value}")
        return [Path(match) for match in matches]
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return [path]


def _parse_csv_args(values: List[str] | None) -> List[str]:
    if not values:
        return []
    items: List[str] = []
    for value in values:
        for row in csv.reader([value], skipinitialspace=True):
            for item in row:
                item = item.strip()
                if item:
                    items.append(item)
    return items


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute a percentile of empty data.")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _summarize_lengths(lengths: Sequence[int]) -> Dict[str, float]:
    sorted_vals = sorted(float(value) for value in lengths)
    q1 = _percentile(sorted_vals, 0.25)
    q3 = _percentile(sorted_vals, 0.75)
    return {
        "count": float(len(sorted_vals)),
        "mean": statistics.fmean(sorted_vals),
        "median": _percentile(sorted_vals, 0.5),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p90": _percentile(sorted_vals, 0.90),
        "p95": _percentile(sorted_vals, 0.95),
        "max": sorted_vals[-1],
    }


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    try:
        from tabulate import tabulate  # type: ignore

        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        widths = [len(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))
        sep = "-+-".join("-" * width for width in widths)
        lines = [
            " | ".join(header.ljust(width) for header, width in zip(headers, widths)),
            sep,
        ]
        for row in rows:
            lines.append(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))
        return "\n".join(lines)


def _batch_token_lengths(
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    progress: bool = False,
) -> List[int]:
    lengths: List[int] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        _log(progress, f"Tokenizing responses {start + 1:,}-{end:,} / {total:,}")
        batch = list(texts[start:end])
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        lengths.extend(len(ids) for ids in input_ids)
    return lengths


def _load_tokenizer(model_name_or_path: str, use_fast: bool, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import guard
        raise RuntimeError("transformers is required for compare_lengths.py") from exc

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # pragma: no cover - runtime import guard
        raise RuntimeError(f"Failed to load tokenizer from '{model_name_or_path}'") from exc


def _default_plot_path(dataset: str | None) -> Path:
    stem = f"{dataset}_response_lengths" if dataset else "response_lengths"
    return Path("figure") / f"{stem}.png"


def _wrap_label(label: str, width: int = 18) -> str:
    pieces = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(pieces) if pieces else label


def _build_palette(count: int) -> List[str]:
    base = [
        "#4E79A7",
        "#59A14F",
        "#B07AA1",
        "#E15759",
        "#76B7B2",
        "#9C755F",
        "#F28E2B",
        "#BAB0AC",
    ]
    return [base[idx % len(base)] for idx in range(count)]


def _plot_lengths(
    path: Path,
    labels: Sequence[str],
    lengths_by_label: Dict[str, List[int]],
    summaries: Dict[str, Dict[str, float]],
    point_sample: int,
    seed: int,
    show: bool,
    title: str | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for compare_lengths.py") from exc

    del point_sample, seed

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "serif"],
            "font.size": 12,
            "axes.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    ordered_data = [lengths_by_label[label] for label in labels]
    if any(not values for values in ordered_data):
        missing = [label for label in labels if not lengths_by_label[label]]
        raise ValueError(f"No response lengths available for: {', '.join(missing)}")

    palette = _build_palette(len(labels))
    fig_height = max(2.8, 0.82 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, fig_height), constrained_layout=True)
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#9AA4B2")

    positions = list(range(len(labels), 0, -1))
    violins = ax.violinplot(
        ordered_data,
        positions=positions,
        vert=False,
        widths=0.84,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, pos, color in zip(violins["bodies"], positions, palette):
        path_obj = body.get_paths()[0]
        vertices = path_obj.vertices
        vertices[vertices[:, 1] < pos, 1] = pos
        body.set_facecolor(mcolors.to_rgba(color, 0.24))
        body.set_edgecolor(color)
        body.set_linewidth(1.0)
        body.set_alpha(1.0)

    box_height = 0.18
    median_half = box_height / 2.0 + 0.02
    for pos, label, color in zip(positions, labels, palette):
        summary = summaries[label]
        q1 = float(summary["q1"])
        q3 = float(summary["q3"])
        median = float(summary["median"])
        width = max(q3 - q1, 1e-9)
        box = Rectangle(
            (q1, pos - box_height / 2.0),
            width,
            box_height,
            facecolor=(1.0, 1.0, 1.0, 0.95),
            edgecolor=color,
            linewidth=1.05,
            zorder=4,
        )
        ax.add_patch(box)
        ax.plot(
            [median, median],
            [pos - median_half, pos + median_half],
            color="#1F2937",
            linewidth=1.2,
            zorder=5,
        )

    all_values = [value for values in ordered_data for value in values]
    x_max = max(all_values)
    x_pad = max(4.0, x_max * 0.025)
    ax.set_xlim(0.0, x_max + x_pad)
    ax.set_ylim(0.45, len(labels) + 0.55)

    ax.set_yticks(positions)
    ax.set_yticklabels([_wrap_label(label) for label in labels])
    ax.set_xlabel("Tokens per response")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(True, axis="x", color="#D7DCE3", linewidth=0.8, alpha=0.55)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", colors="#1F2937")
    ax.tick_params(axis="y", length=0, pad=8, colors="#1F2937")

    if title:
        ax.set_title(title, loc="left", pad=8, fontsize=14)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-response token lengths across JSONL generation outputs. "
            "Supports the same --inputs or --dataset/--tags grouping style as compare_trajs.py."
        )
    )
    parser.add_argument(
        "--model",
        "--model-name-or-path",
        dest="model_name_or_path",
        required=True,
        help="Tokenizer source passed to transformers.AutoTokenizer.from_pretrained().",
    )
    parser.add_argument(
        "--inputs",
        action="append",
        help="Comma-separated list of JSONL paths/globs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--labels",
        action="append",
        help="Comma-separated labels for inputs/tags (same order as groups).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name to expand --tags into ckpt/{dataset}/{tag}/*.jsonl.",
    )
    parser.add_argument(
        "--tags",
        action="append",
        help="Comma-separated tag names to compare (requires --dataset).",
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root directory for --dataset/--tags expansion (default: ckpt).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Output path for the figure (default: figure/{dataset}_response_lengths.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively after saving it.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenizer batch size (default: 256).",
    )
    parser.add_argument(
        "--point-sample",
        type=int,
        default=900,
        help="Accepted for CLI compatibility; unused by the paper-style renderer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Accepted for CLI compatibility; unused by the paper-style renderer.",
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Load the slow tokenizer implementation instead of the fast one.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print loading/tokenization progress to stderr.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.point_sample < 0:
        raise ValueError("--point-sample must be >= 0.")

    input_specs = _parse_csv_args(args.inputs)
    input_groups: List[Dict[str, str]] = []
    for spec in input_specs:
        input_groups.append({"spec": spec, "label": spec})

    tag_specs = _parse_csv_args(args.tags)
    if tag_specs:
        if not args.dataset:
            raise ValueError("--dataset is required when using --tags.")
        for tag in tag_specs:
            input_groups.append(
                {
                    "spec": os.path.join(args.ckpt_root, args.dataset, tag, "*.jsonl"),
                    "label": tag,
                }
            )

    if not input_groups:
        raise RuntimeError("No inputs provided. Use --inputs or --dataset/--tags.")

    label_specs = _parse_csv_args(args.labels)
    if label_specs and len(label_specs) != len(input_groups):
        raise ValueError("Number of labels must match number of input groups.")
    if label_specs:
        for group, label in zip(input_groups, label_specs):
            group["label"] = label
    labels_candidate = [group["label"] for group in input_groups]
    if len(set(labels_candidate)) != len(labels_candidate):
        raise ValueError("Labels must be unique.")

    tokenizer = _load_tokenizer(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    labels: List[str] = []
    lengths_by_label: Dict[str, List[int]] = {}
    file_counts: Dict[str, int] = {}
    summaries: Dict[str, Dict[str, float]] = {}

    for idx, group in enumerate(input_groups, 1):
        spec = group["spec"]
        label = group["label"]
        labels.append(label)
        paths = _resolve_inputs(spec)
        file_counts[label] = len(paths)
        _log(args.progress, f"[{idx}/{len(input_groups)}] Collecting responses for {label}")
        responses = _load_responses(paths, progress=args.progress)
        if not responses:
            raise RuntimeError(f"No responses found for '{label}' from spec: {spec}")
        lengths = _batch_token_lengths(
            tokenizer,
            responses,
            batch_size=args.batch_size,
            progress=args.progress,
        )
        lengths_by_label[label] = lengths
        summaries[label] = _summarize_lengths(lengths)

    rows: List[List[str]] = []
    for label in labels:
        summary = summaries[label]
        rows.append(
            [
                label,
                f"{file_counts[label]:,}",
                f"{int(summary['count']):,}",
                f"{summary['mean']:.1f}",
                f"{summary['median']:.1f}",
                f"{summary['iqr']:.1f}",
                f"{summary['p90']:.1f}",
                f"{summary['max']:.0f}",
            ]
        )

    print(
        _format_table(
            ["Method", "Files", "Responses", "Mean", "Median", "IQR", "P90", "Max"],
            rows,
        )
    )

    plot_path = (
        Path(os.path.expanduser(args.plot))
        if args.plot
        else _default_plot_path(args.dataset)
    )
    _plot_lengths(
        path=plot_path,
        labels=labels,
        lengths_by_label=lengths_by_label,
        summaries=summaries,
        point_sample=args.point_sample,
        seed=args.seed,
        show=args.show,
        title=args.title,
    )
    print(f"\nSaved figure to {plot_path}")


if __name__ == "__main__":
    main()
