"""Compare two JSONL trajectory files with maj@k/pass@k metrics."""
from __future__ import annotations

import argparse
import json
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


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


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


def _compute_metrics(
    scores_a: Dict[str, List[float]],
    scores_b: Dict[str, List[float]],
    configs: List[Tuple[str, int]],
) -> List[List[str]]:
    common = sorted(set(scores_a) & set(scores_b))
    rows: List[List[str]] = []
    for mode, k in configs:
        used = 0
        correct_a = 0
        correct_b = 0
        for pid in common:
            sa = scores_a[pid]
            sb = scores_b[pid]
            if len(sa) < k or len(sb) < k:
                continue
            used += 1
            a_slice = sa[:k]
            b_slice = sb[:k]
            if mode == "maj":
                a_ok = (sum(a_slice) / k) > 0.5
                b_ok = (sum(b_slice) / k) > 0.5
            else:
                a_ok = max(a_slice) > 0.5
                b_ok = max(b_slice) > 0.5
            correct_a += 1 if a_ok else 0
            correct_b += 1 if b_ok else 0
        acc_a = (correct_a / used * 100.0) if used > 0 else None
        acc_b = (correct_b / used * 100.0) if used > 0 else None
        delta = (acc_b - acc_a) if (acc_a is not None and acc_b is not None) else None
        label = f"{mode}@{k}"
        rows.append([label, _format_pct(acc_a), _format_pct(acc_b), _format_pct(delta), str(used)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two JSONL trajectory files with maj@k / pass@k metrics."
    )
    parser.add_argument("--input_a", required=True, help="Path to first JSONL.")
    parser.add_argument("--input_b", required=True, help="Path to second JSONL.")
    parser.add_argument(
        "--config",
        action="append",
        help="Metric config(s) like maj4 or pass@2. Can be repeated or comma-separated.",
    )
    parser.add_argument("--label_a", default=None, help="Label for input A column.")
    parser.add_argument("--label_b", default=None, help="Label for input B column.")
    args = parser.parse_args()

    path_a = Path(args.input_a)
    path_b = Path(args.input_b)
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError("Both --input_a and --input_b must exist.")

    scores_a = _load_scores(path_a)
    scores_b = _load_scores(path_b)
    if not scores_a or not scores_b:
        raise RuntimeError("Failed to load scores from one or both inputs.")

    common = set(scores_a) & set(scores_b)
    if not common:
        raise RuntimeError("No overlapping problems between the two inputs.")

    min_samples = min(
        min(len(scores_a[pid]), len(scores_b[pid])) for pid in common
    )
    if min_samples < 1:
        raise RuntimeError("Insufficient samples to compute metrics.")

    configs = _parse_configs(args.config, max_k=min_samples)
    if not configs:
        configs = [("maj", k) for k in range(1, min_samples + 1)]
        configs += [("pass", k) for k in range(1, min_samples + 1)]

    label_a = args.label_a or path_a.name
    label_b = args.label_b or path_b.name
    headers = ["Metric", label_a, label_b, "Delta(B-A)", "Problems"]

    rows = _compute_metrics(scores_a, scores_b, configs)
    table = _format_table(headers, rows)
    print(table)


if __name__ == "__main__":
    main()
