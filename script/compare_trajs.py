"""Compare JSONL trajectory files with maj@k/pass@k metrics (paper-style estimator)."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS_PATH = _REPO_ROOT / "utils"
if str(_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(_UTILS_PATH))

try:
    from boxed_extract import compute_score  # type: ignore
except Exception as exc:  # pragma: no cover - runtime import guard
    raise RuntimeError("Failed to import compute_score from utils/boxed_extract.py") from exc


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


def _extract_problem_id(row: Dict[str, Any], fallback_idx: int) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("problem_index", "problem_id", "idx", "index"):
        if key in meta:
            return f"{meta[key]}"
        if key in row:
            return f"{row[key]}"
    for key in ("problem", "prompt"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return f"{key}:{row[key]}"
    return f"row:{fallback_idx}"


def _extract_sample_index(row: Dict[str, Any], fallback_idx: int) -> int:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("sample_index", "sample_id", "sample_idx"):
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                break
        if key in row:
            try:
                return int(row[key])
            except Exception:
                break
    return fallback_idx


def _extract_score(row: Dict[str, Any]) -> float | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if "score" in meta:
        try:
            return float(meta["score"])
        except Exception:
            return None
    if "score" in row:
        try:
            return float(row["score"])
        except Exception:
            return None
    response = None
    for key in ("response", "completion", "solution", "output", "generated", "answer"):
        if key in row and isinstance(row[key], str):
            response = row[key]
            break
    ground_truth = None
    for key in ("ground_truth", "gt", "answer"):
        if key in meta:
            ground_truth = meta[key]
            break
        if key in row:
            ground_truth = row[key]
            break
    if response is not None and ground_truth is not None:
        try:
            return float(compute_score(response, ground_truth))
        except Exception:
            return None
    return None


def _load_scores(path: Path) -> Dict[str, List[float]]:
    problems: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    per_problem_counter: Dict[str, int] = defaultdict(int)

    for row_idx, row in enumerate(_read_jsonl(path)):
        if isinstance(row.get("solutions"), list) and row.get("ground_truth") is not None:
            problem_id = _extract_problem_id(row, row_idx)
            ground_truth = row.get("ground_truth")
            scores_list = row.get("scores")
            for sol_idx, solution in enumerate(row["solutions"]):
                score = None
                if isinstance(scores_list, list) and sol_idx < len(scores_list):
                    try:
                        score = float(scores_list[sol_idx])
                    except Exception:
                        score = None
                if score is None:
                    try:
                        score = float(compute_score(str(solution), ground_truth))
                    except Exception:
                        continue
                problems[problem_id].append((sol_idx, sol_idx, score))
            continue

        problem_id = _extract_problem_id(row, row_idx)
        sample_idx_default = per_problem_counter[problem_id]
        sample_idx = _extract_sample_index(row, sample_idx_default)
        per_problem_counter[problem_id] += 1
        score = _extract_score(row)
        if score is None:
            continue
        problems[problem_id].append((sample_idx, sample_idx_default, score))

    ordered_scores: Dict[str, List[float]] = {}
    for problem_id, triples in problems.items():
        triples_sorted = sorted(triples, key=lambda t: (t[0], t[1]))
        ordered_scores[problem_id] = [float(score) for _, _, score in triples_sorted]
    return ordered_scores


def _merge_scores(paths: List[Path]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = defaultdict(list)
    for path in paths:
        scores = _load_scores(path)
        for pid, vals in scores.items():
            merged[pid].extend(vals)
    return dict(merged)


def _parse_configs(configs: List[str] | None, max_k: int) -> List[Tuple[str, int]]:
    if not configs:
        return []
    parsed: List[Tuple[str, int]] = []
    for raw in configs:
        for item in raw.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item.startswith("maj"):
                mode = "maj"
                suffix = item.replace("maj", "", 1)
            elif item.startswith("pass"):
                mode = "pass"
                suffix = item.replace("pass", "", 1)
            else:
                raise ValueError(f"Unknown config '{item}'. Use majK or passK.")
            if suffix.startswith("@"):
                suffix = suffix[1:]
            if not suffix.isdigit():
                raise ValueError(f"Invalid config '{item}'. Use majK or passK.")
            k = int(suffix)
            if k < 1 or k > max_k:
                raise ValueError(f"Config '{item}' out of range (1..{max_k}).")
            parsed.append((mode, k))
    return parsed


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


def _t_critical_975(df: int) -> float:
    if df <= 0:
        return 0.0
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(df, 1.96)


def _format_mean_ci(mean: float | None, ci: float | None) -> str:
    if mean is None:
        return "n/a"
    if ci is None:
        return f"{mean:.2f}%"
    return f"{mean:.2f}±{ci:.2f}%"


def _format_diff_ci(mean: float | None, ci: float | None) -> str:
    if mean is None:
        return "n/a"
    if ci is None:
        return f"{mean:+.2f}%"
    return f"{mean:+.2f}±{ci:.2f}%"


def _resolve_focus_index(labels: List[str], focus: str | None) -> int | None:
    if focus is None:
        return None
    focus = focus.strip()
    if not focus or focus.lower() in {"none", "off", "false"}:
        return None
    if focus.lower() == "auto":
        return 0 if labels else None
    if focus.isdigit():
        idx = int(focus) - 1
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"--focus index out of range: {focus}")
        return idx
    if focus in labels:
        return labels.index(focus)
    lowered = [label.lower() for label in labels]
    focus_lower = focus.lower()
    if focus_lower in lowered:
        return lowered.index(focus_lower)
    matches = [i for i, label in enumerate(lowered) if focus_lower in label]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"--focus '{focus}' did not match labels: {', '.join(labels)}")


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def fmt_row(parts: List[str]) -> str:
            return "| " + " | ".join(part.ljust(widths[i]) for i, part in enumerate(parts)) + " |"

        border = "|-" + "-|-".join("-" * w for w in widths) + "-|"
        out = [fmt_row(headers), border]
        out.extend(fmt_row(row) for row in rows)
        return "\n".join(out)


def _count_correct(scores: List[float]) -> int:
    return sum(1 for s in scores if s > 0.5)


def _pass_at_k_estimate(n: int, c: int, k: int) -> float | None:
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _maj_at_k_estimate(n: int, c: int, k: int) -> float | None:
    if n < k:
        return None
    total = math.comb(n, k)
    needed = k // 2 + 1
    prob = 0.0
    for j in range(needed, k + 1):
        if j > c:
            break
        if k - j > n - c:
            continue
        prob += (math.comb(c, j) * math.comb(n - c, k - j)) / total
    return prob


def _metric_for_scores(
    scores: Dict[str, List[float]],
    common: List[str],
    mode: str,
    k: int,
) -> Tuple[float | None, int]:
    values: List[float] = []
    for pid in common:
        vals = scores.get(pid)
        if not vals:
            continue
        n = len(vals)
        c = _count_correct(vals)
        if mode == "pass":
            est = _pass_at_k_estimate(n, c, k)
        else:
            est = _maj_at_k_estimate(n, c, k)
        if est is None:
            continue
        values.append(est)
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _seed_metrics(
    scores_list: List[Dict[str, List[float]]],
    common: List[str],
    mode: str,
    k: int,
) -> Tuple[List[float], List[int]]:
    metrics: List[float] = []
    used_counts: List[int] = []
    for scores in scores_list:
        mean_val, used = _metric_for_scores(scores, common, mode, k)
        if mean_val is None:
            continue
        metrics.append(mean_val)
        used_counts.append(used)
    return metrics, used_counts


def _per_seed_metrics(
    scores_list: List[Dict[str, List[float]]],
    common: List[str],
    mode: str,
    k: int,
) -> List[float | None]:
    metrics: List[float | None] = []
    for scores in scores_list:
        mean_val, _ = _metric_for_scores(scores, common, mode, k)
        metrics.append(mean_val)
    return metrics


def _compute_metrics(
    groups: List[Tuple[str, List[Dict[str, List[float]]], Dict[str, List[float]]]],
    configs: List[Tuple[str, int]],
    common: List[str],
) -> Tuple[List[List[str]], Dict[str, List[List[Tuple[int, float, float | None]]]]]:
    rows: List[List[str]] = []
    plot_data: Dict[str, List[List[Tuple[int, float, float | None]]]] = {
        "maj": [[] for _ in groups],
        "pass": [[] for _ in groups],
    }
    for mode, k in configs:
        row: List[str] = [f"{mode}@{k}"]
        used_counts: List[int] = []
        means: List[float | None] = []
        cis: List[float | None] = []

        for _, scores_list, pooled in groups:
            pooled_mean, used = _metric_for_scores(pooled, common, mode, k)
            seed_vals, used_list = _seed_metrics(scores_list, common, mode, k)
            seed_mean = sum(seed_vals) / len(seed_vals) if seed_vals else None
            if len(seed_vals) > 1:
                stdev = statistics.stdev(seed_vals)
                t_critical = _t_critical_975(len(seed_vals) - 1)
                ci = t_critical * stdev / math.sqrt(len(seed_vals))
            else:
                ci = None
            mean = seed_mean if seed_mean is not None else pooled_mean
            means.append(mean * 100.0 if mean is not None else None)
            cis.append(ci * 100.0 if ci is not None else None)
            used_counts.append(used if used else (min(used_list) if used_list else 0))

        for mean, ci in zip(means, cis):
            row.append(_format_mean_ci(mean, ci))

        used = min(u for u in used_counts if u > 0) if any(u > 0 for u in used_counts) else 0
        row.append(str(used))
        rows.append(row)

        for idx, (mean, ci) in enumerate(zip(means, cis)):
            if mean is not None:
                plot_data[mode][idx].append((k, mean, ci))
    return rows, plot_data


def _compute_pairwise_diffs(
    groups: List[Tuple[str, List[Dict[str, List[float]]], Dict[str, List[float]]]],
    configs: List[Tuple[str, int]],
    common: List[str],
    labels: List[str],
) -> List[Tuple[str, str]]:
    tables: List[Tuple[str, str]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            label_i = labels[i]
            label_j = labels[j]
            rows: List[List[str]] = []
            for mode, k in configs:
                seeds_i = _per_seed_metrics(groups[i][1], common, mode, k)
                seeds_j = _per_seed_metrics(groups[j][1], common, mode, k)
                diffs = [
                    a - b
                    for a, b in zip(seeds_i, seeds_j)
                    if a is not None and b is not None
                ]
                if not diffs:
                    mean = None
                    ci = None
                else:
                    mean = sum(diffs) / len(diffs)
                    if len(diffs) > 1:
                        stdev = statistics.stdev(diffs)
                        t_critical = _t_critical_975(len(diffs) - 1)
                        ci = t_critical * stdev / math.sqrt(len(diffs))
                    else:
                        ci = None
                mean_pct = mean * 100.0 if mean is not None else None
                ci_pct = ci * 100.0 if ci is not None else None
                sig = (
                    "yes"
                    if mean is not None and ci is not None and (mean - ci > 0 or mean + ci < 0)
                    else "no"
                )
                rows.append(
                    [
                        f"{mode}@{k}",
                        _format_diff_ci(mean_pct, ci_pct),
                        str(len(diffs)),
                        sig,
                    ]
                )
            headers = ["Metric", f"{label_i}-{label_j}", "Seeds", "Sig95"]
            tables.append((f"{label_i} vs {label_j}", _format_table(headers, rows)))
    return tables


def _plot_results(
    path: Path,
    plot_data: Dict[str, List[List[Tuple[int, float, float | None]]]],
    labels: List[str],
    focus_idx: int | None,
    maj_avg: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot output") from exc

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "serif"],
            "font.size": 12,
            "axes.titleweight": "bold",
        }
    )

    if plot_data.get("maj") and len(plot_data["maj"]) != len(labels):
        raise ValueError("Plot data does not match number of labels.")

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True, constrained_layout=True)
    order = [("maj", "maj@k (%)"), ("pass", "pass@k (%)")]

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    non_focus_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    focus_color = "#004488"
    non_focus_palette = [
        "#EE7733",
        "#44AA99",
        "#66CCEE",
        "#999933",
        "#BBBBBB",
    ]

    def _pairwise_average(series: List[Tuple[int, float, float | None]]) -> List[Tuple[int, float, float | None]]:
        by_k = {k: (v, s) for k, v, s in series}
        averaged: List[Tuple[int, float, float | None]] = []
        for k in sorted(by_k):
            if k % 2 == 0 and (k - 1) in by_k:
                v_even, s_even = by_k[k]
                v_odd, s_odd = by_k[k - 1]
                avg_val = (v_even + v_odd) / 2.0
                if s_even is None or s_odd is None:
                    avg_ci = None
                else:
                    avg_ci = (s_even + s_odd) / 2.0
                averaged.append((k, avg_val, avg_ci))
        return averaged

    for ax, (mode, ylabel) in zip(axes, order):
        series_groups = plot_data.get(mode, [])
        if not series_groups:
            ax.text(
                0.5,
                0.5,
                (
                    f"No data for {mode}@k"
                    if mode == "maj"
                    else f"No data for {mode}@k"
                ),
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        has_any = False

        total_series = len(series_groups)
        non_focus_total = (
            total_series - 1 if focus_idx is not None and total_series > 0 else 0
        )
        for idx, series in enumerate(series_groups):
            if mode == "maj":
                if maj_avg == "pairs":
                    series = _pairwise_average(series)
                elif maj_avg == "odd":
                    series = [(k, v, s) for (k, v, s) in series if k % 2 == 1]
            if not series:
                continue
            series.sort(key=lambda t: t[0])
            ks = [k for k, _, _ in series]
            vals = [v for _, v, _ in series]
            color_cycle = f"C{idx % 10}"
            is_focus = focus_idx is None or idx == focus_idx
            if focus_idx is None:
                line_color = color_cycle
                line_alpha = 0.9
                line_z = 3
                line_style = "-"
                marker = None
                markersize = 0
            elif is_focus:
                line_color = focus_color
                line_alpha = 0.95
                line_z = 3
                line_style = "-"
                marker = "o"
                markersize = 4
            else:
                rank = idx if focus_idx is None else sum(1 for j in range(idx) if j != focus_idx)
                line_color = (
                    non_focus_palette[rank]
                    if rank < len(non_focus_palette)
                    else non_focus_palette[-1]
                )
                line_alpha = 0.8
                line_z = 2
                line_style = non_focus_styles[rank % len(non_focus_styles)]
                marker = None
                markersize = 0

            ax.plot(
                ks,
                vals,
                color=line_color,
                linewidth=1.6,
                alpha=line_alpha,
                label=labels[idx],
                zorder=line_z,
                linestyle=line_style,
                marker=marker,
                markersize=markersize,
            )
            has_any = True

        if not has_any:
            ax.text(
                0.5,
                0.5,
                (
                    f"No data for {mode}@k"
                    if mode == "maj"
                    else f"No data for {mode}@k"
                ),
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if maj_avg == "pairs":
        xlabel = "k (maj: pairwise avg of k-1,k at even k)"
    elif maj_avg == "odd":
        xlabel = "k (maj: odd only)"
    else:
        xlabel = "k"
    axes[-1].set_xlabel(xlabel)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    fig.suptitle("Trajectory Comparison", fontsize=14, y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_inputs(value: str) -> List[Path]:
    expanded = os.path.expanduser(value)
    if glob.has_magic(expanded):
        matches = sorted(glob.glob(expanded))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {value}")
        return [Path(m) for m in matches]
    path = Path(expanded)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {value}")
    return [path]


def _sample_paths(paths: List[Path], count: int, seed: int | None) -> List[Path]:
    if count >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    chosen = rng.sample(paths, count)
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple JSONL trajectory files with maj@k / pass@k metrics. "
            "Uses a hypergeometric estimator over all samples and reports mean±95% CI "
            "across seeds when multiple files are provided."
        )
    )
    parser.add_argument(
        "--inputs",
        action="append",
        help=(
            "Comma-separated list of JSONL paths/globs. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for subsampling when glob patterns yield different counts.",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Metric config(s) like maj4 or pass@2. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional output path for a PDF/PNG plot.",
    )
    parser.add_argument(
        "--maj-avg",
        choices=("pairs", "odd", "all"),
        default="pairs",
        help=(
            "Plotting mode for maj@k: 'pairs' averages (2i-1,2i) at even k; "
            "'odd' shows odd k only; 'all' shows every k (default: pairs)."
        ),
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help=(
            "Report pairwise differences across seeds with 95% CI. "
            "Differences are paired by seed index (same order as inputs)."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=64,
        help="Maximum k to compute (default: 64).",
    )
    parser.add_argument(
        "--labels",
        action="append",
        help="Comma-separated labels for inputs (same order as --inputs).",
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root directory for --dataset/--tags expansion (default: ckpt).",
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
        "--focus",
        default="none",
        help=(
            "Label or 1-based index to emphasize in plots; others are muted. "
            "Use 'none' to disable (default: none)."
        ),
    )
    args = parser.parse_args()

    input_specs = _parse_csv_args(args.inputs)
    input_labels = list(input_specs)

    tag_specs = _parse_csv_args(args.tags)
    if tag_specs:
        if not args.dataset:
            raise ValueError("--dataset is required when using --tags.")
        for tag in tag_specs:
            spec = os.path.join(args.ckpt_root, args.dataset, tag, "*.jsonl")
            input_specs.append(spec)
            input_labels.append(tag)

    if not input_specs:
        raise RuntimeError("No inputs provided. Use --inputs or --dataset/--tags.")

    label_specs = _parse_csv_args(args.labels)
    if label_specs and len(label_specs) != len(input_specs):
        raise ValueError("Number of labels must match number of inputs.")
    labels = label_specs if label_specs else input_labels
    focus_idx = _resolve_focus_index(labels, args.focus)

    groups_raw: List[Tuple[str, List[Path]]] = []
    for spec in input_specs:
        groups_raw.append((spec, _resolve_inputs(spec)))

    counts = [len(paths) for _, paths in groups_raw]
    if len(set(counts)) > 1:
        min_seeds = min(counts)
        counts_str = ", ".join(
            f"{label}={count}" for label, count in zip(labels, counts)
        )
        prompt = (
            f"Found differing counts ({counts_str}). "
            f"Use {min_seeds} seeds by randomly subsampling larger sets? [y/N]: "
        )
        response = input(prompt).strip().lower()
        if response not in ("y", "yes"):
            print("Aborting without computing metrics.")
            sys.exit(1)
        new_groups_raw: List[Tuple[str, List[Path]]] = []
        for (spec, paths), label in zip(groups_raw, labels):
            if len(paths) > min_seeds:
                paths = _sample_paths(paths, min_seeds, args.seed)
                print(f"Subsampled {label} to {min_seeds} files:")
                for path in paths:
                    print(f"  {path}")
            new_groups_raw.append((spec, paths))
        groups_raw = new_groups_raw

    groups: List[Tuple[str, List[Dict[str, List[float]]], Dict[str, List[float]]]] = []
    for label, paths in zip(labels, [paths for _, paths in groups_raw]):
        scores_list = [_load_scores(p) for p in paths]
        pooled = _merge_scores(paths)
        if not pooled:
            raise RuntimeError(f"Failed to load scores for {label}.")
        groups.append((label, scores_list, pooled))

    all_dicts: List[Dict[str, List[float]]] = []
    for _, scores_list, _ in groups:
        all_dicts.extend(scores_list)
    if not all_dicts:
        raise RuntimeError("No scores loaded from inputs.")
    common = set(all_dicts[0])
    for scores in all_dicts[1:]:
        common &= set(scores)
    if not common:
        raise RuntimeError("No overlapping problems between the two inputs.")

    min_samples_pooled = min(
        min(len(group[2][pid]) for group in groups) for pid in common
    )
    if min_samples_pooled < 1:
        raise RuntimeError("Insufficient samples to compute metrics.")

    if args.max_k < 1:
        raise ValueError("--max-k must be >= 1.")
    effective_max_k = min(args.max_k, min_samples_pooled)
    configs = _parse_configs(args.config, max_k=effective_max_k)
    if not configs:
        configs = [("maj", k) for k in range(1, effective_max_k + 1)]
        configs += [("pass", k) for k in range(1, effective_max_k + 1)]

    headers = ["Metric"] + labels + ["Problems"]

    rows, plot_data = _compute_metrics(groups, configs, sorted(common))
    table = _format_table(headers, rows)
    print(table)
    if args.diff:
        diff_tables = _compute_pairwise_diffs(groups, configs, sorted(common), labels)
        for title, diff_table in diff_tables:
            print()
            print(title)
            print(diff_table)
    if args.plot:
        _plot_results(Path(args.plot), plot_data, labels, focus_idx, args.maj_avg)


if __name__ == "__main__":
    main()
