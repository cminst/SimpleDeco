"""Compare two JSONL trajectory files with maj@k/pass@k metrics (paper-style estimator)."""
from __future__ import annotations

import argparse
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


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.2f}%"
    return f"{mean:.2f}±{std:.2f}%"


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


def _compute_metrics(
    scores_a_list: List[Dict[str, List[float]]],
    scores_b_list: List[Dict[str, List[float]]],
    pooled_a: Dict[str, List[float]],
    pooled_b: Dict[str, List[float]],
    configs: List[Tuple[str, int]],
    common: List[str],
) -> Tuple[List[List[str]], Dict[str, List[Tuple[int, float, float]]]]:
    rows: List[List[str]] = []
    plot_data: Dict[str, List[Tuple[int, float, float]]] = {"maj": [], "pass": []}
    for mode, k in configs:
        pooled_mean_a, used_a = _metric_for_scores(pooled_a, common, mode, k)
        pooled_mean_b, used_b = _metric_for_scores(pooled_b, common, mode, k)

        seed_vals_a, used_list_a = _seed_metrics(scores_a_list, common, mode, k)
        seed_vals_b, used_list_b = _seed_metrics(scores_b_list, common, mode, k)

        std_a = None
        std_b = None
        if len(seed_vals_a) > 1:
            std_a = statistics.pstdev(seed_vals_a) * 100.0
        if len(seed_vals_b) > 1:
            std_b = statistics.pstdev(seed_vals_b) * 100.0

        mean_a = pooled_mean_a * 100.0 if pooled_mean_a is not None else None
        mean_b = pooled_mean_b * 100.0 if pooled_mean_b is not None else None
        delta = (mean_b - mean_a) if (mean_a is not None and mean_b is not None) else None

        used = min(used_a, used_b) if used_a and used_b else 0
        if not used and (used_list_a or used_list_b):
            used = min(used_list_a + used_list_b)

        label = f"{mode}@{k}"
        rows.append(
            [
                label,
                _format_mean_std(mean_a, std_a),
                _format_mean_std(mean_b, std_b),
                _format_mean_std(delta, None),
                str(used),
            ]
        )
        if mean_a is not None and mean_b is not None:
            plot_data[mode].append((k, mean_a, mean_b))
    return rows, plot_data


def _plot_results(
    path: Path,
    plot_data: Dict[str, List[Tuple[int, float, float]]],
    label_a: str,
    label_b: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot output") from exc

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Times", "serif"],
            "font.size": 12,
            "axes.titleweight": "bold",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True, constrained_layout=True)
    order = [("maj", "maj@k (%)"), ("pass", "pass@k (%)")]

    for ax, (mode, ylabel) in zip(axes, order):
        series = [(k, a, b) for (k, a, b) in plot_data.get(mode, []) if k % 2 == 1]
        if not series:
            ax.text(
                0.5,
                0.5,
                f"No odd k data for {mode}@k",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        series.sort(key=lambda t: t[0])
        ks = [k for k, _, _ in series]
        vals_a = [a for _, a, _ in series]
        vals_b = [b for _, _, b in series]

        ax.plot(ks, vals_a, marker="o", linewidth=2.0, markersize=4, label=label_a)
        ax.plot(ks, vals_b, marker="s", linewidth=2.0, markersize=4, label=label_b)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    axes[-1].set_xlabel("k (odd only)")
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
            "Compare JSONL trajectory files with maj@k / pass@k metrics. "
            "Uses a hypergeometric estimator over all samples and reports mean±std "
            "across seeds when multiple files are provided."
        )
    )
    parser.add_argument(
        "--input_a", required=True, help="Path or glob pattern to first JSONL(s)."
    )
    parser.add_argument(
        "--input_b", required=True, help="Path or glob pattern to second JSONL(s)."
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
        help="Optional output path for a PDF/PNG plot (odd k only).",
    )
    parser.add_argument("--label_a", default=None, help="Label for input A column.")
    parser.add_argument("--label_b", default=None, help="Label for input B column.")
    args = parser.parse_args()

    paths_a = _resolve_inputs(args.input_a)
    paths_b = _resolve_inputs(args.input_b)

    if len(paths_a) != len(paths_b):
        min_seeds = min(len(paths_a), len(paths_b))
        prompt = (
            f"Found {len(paths_a)} files for A and {len(paths_b)} for B. "
            f"Use {min_seeds} seeds by randomly subsampling the larger set? [y/N]: "
        )
        response = input(prompt).strip().lower()
        if response not in ("y", "yes"):
            print("Aborting without computing metrics.")
            sys.exit(1)
        if len(paths_a) > min_seeds:
            paths_a = _sample_paths(paths_a, min_seeds, args.seed)
            print(f"Subsampled A to {min_seeds} files:")
            for path in paths_a:
                print(f"  {path}")
        if len(paths_b) > min_seeds:
            paths_b = _sample_paths(paths_b, min_seeds, args.seed)
            print(f"Subsampled B to {min_seeds} files:")
            for path in paths_b:
                print(f"  {path}")

    scores_a_list = [_load_scores(p) for p in paths_a]
    scores_b_list = [_load_scores(p) for p in paths_b]
    pooled_a = _merge_scores(paths_a)
    pooled_b = _merge_scores(paths_b)
    if not pooled_a or not pooled_b:
        raise RuntimeError("Failed to load scores from one or both inputs.")

    all_dicts = scores_a_list + scores_b_list
    if not all_dicts:
        raise RuntimeError("No scores loaded from inputs.")
    common = set(all_dicts[0])
    for scores in all_dicts[1:]:
        common &= set(scores)
    if not common:
        raise RuntimeError("No overlapping problems between the two inputs.")

    min_samples_pooled = min(
        min(len(pooled_a[pid]), len(pooled_b[pid])) for pid in common
    )
    if min_samples_pooled < 1:
        raise RuntimeError("Insufficient samples to compute metrics.")

    configs = _parse_configs(args.config, max_k=min_samples_pooled)
    if not configs:
        configs = [("maj", k) for k in range(1, min_samples_pooled + 1)]
        configs += [("pass", k) for k in range(1, min_samples_pooled + 1)]

    label_a = args.label_a or args.input_a
    label_b = args.label_b or args.input_b
    headers = ["Metric", label_a, label_b, "Delta(B-A)", "Problems"]

    rows, plot_data = _compute_metrics(
        scores_a_list,
        scores_b_list,
        pooled_a,
        pooled_b,
        configs,
        sorted(common),
    )
    table = _format_table(headers, rows)
    print(table)
    if args.plot:
        _plot_results(Path(args.plot), plot_data, label_a, label_b)


if __name__ == "__main__":
    main()
