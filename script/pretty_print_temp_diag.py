#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty print AutoDeco temperature diagnostics JSON files."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a diagnostics JSON file or directory containing step_*.json files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If --path is a directory, print all diagnostic files (sorted by step).",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="If --path is a directory and --all is not set, print this many latest files (default: 1).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="How many top-token entries to show per distribution (default: 5).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=300,
        help="Maximum number of context characters to print (default: 300).",
    )
    return parser.parse_args()


def _extract_step(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _fmt_token_entry(entry: dict[str, Any]) -> str:
    token_id = entry.get("token_id")
    token = entry.get("token", "")
    prob = entry.get("prob")
    if isinstance(prob, float):
        prob_s = f"{prob:.6f}"
    else:
        prob_s = str(prob)
    token = token.replace("\n", "\\n")
    if token == "":
        token = "<empty>"
    return f"id={token_id}, prob={prob_s}, token={token!r}"


def _select_files(path: Path, print_all: bool, num_files: int) -> list[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = sorted(
        [p for p in path.glob("step_*.json") if p.is_file()],
        key=lambda p: (_extract_step(p), p.name),
    )
    if not files:
        raise FileNotFoundError(f"No step_*.json files found in: {path}")

    if print_all:
        return files

    n = max(1, int(num_files))
    return files[-n:]


def print_diag(path: Path, payload: dict[str, Any], topk: int, max_context_chars: int) -> None:
    step = payload.get("global_step")
    objective = payload.get("temp_objective")
    print("=" * 100)
    print(f"File: {path}")
    print(f"Step: {step} | Objective: {objective}")
    print(
        "Temp targets: "
        f"cap={payload.get('goldilocks_temp_cap')} "
        f"uniform={payload.get('goldilocks_uniform')} "
        f"bins={payload.get('goldilocks_uniform_bins')} "
        f"smooth_window={payload.get('temp_target_smooth_window')}"
    )
    print(
        "Temp weights: "
        f"hinge={payload.get('temp_hinge_weight')} reg={payload.get('temp_reg_weight')}"
    )
    summary = payload.get("target_distribution_summary") or {}
    if summary:
        print(
            "Target dist: "
            f"n={summary.get('count')} mean={summary.get('mean'):.3f} "
            f"std={summary.get('std'):.3f} p10={summary.get('p10'):.3f} "
            f"p50={summary.get('p50'):.3f} p90={summary.get('p90'):.3f}"
        )

    examples = payload.get("examples", [])
    if not isinstance(examples, list) or len(examples) == 0:
        print("No examples in this payload.")
        return

    for i, ex in enumerate(examples, start=1):
        gt = ex.get("ground_truth", {})
        pred = ex.get("prediction", {})
        align = ex.get("min_p_alignment", {})
        context = _truncate(str(ex.get("context_text", "")), max_context_chars)
        gt_token = str(gt.get("token", "")).replace("\n", "\\n")
        print("-" * 100)
        print(
            f"Example {i} | batch_index={ex.get('batch_index')} "
            f"token_pos={ex.get('token_position')} target_pos={ex.get('target_token_position')}"
        )
        print(f"Context: {context}")
        print(
            "Ground truth: "
            f"id={gt.get('token_id')} token={gt_token!r} "
            f"rank_unscaled={gt.get('rank_unscaled')} "
            f"p_unscaled={gt.get('prob_unscaled')} p_at_pred_temp={gt.get('prob_at_pred_temp')}"
        )
        print(
            "Temperature: "
            f"pred={pred.get('predicted_temperature')} required={align.get('required_temperature')} "
            f"gap(required-pred)={align.get('hinge_gap_required_minus_pred')} "
            f"min_p_ok={align.get('condition_satisfied')}"
        )
        print(
            "Min-P details: "
            f"ratio={align.get('min_p_ratio')} "
            f"max_prob_at_pred={align.get('max_prob_at_pred_temp')} "
            f"threshold={align.get('threshold_prob_min_p_times_max')}"
        )

        unscaled = ex.get("top_tokens_unscaled", [])
        scaled = ex.get("top_tokens_at_pred_temp", [])

        print(f"Top tokens (unscaled, up to {topk}):")
        for item in unscaled[:topk]:
            if isinstance(item, dict):
                print(f"  - {_fmt_token_entry(item)}")
        print(f"Top tokens (at predicted temp, up to {topk}):")
        for item in scaled[:topk]:
            if isinstance(item, dict):
                print(f"  - {_fmt_token_entry(item)}")


def main() -> None:
    args = parse_args()
    path = Path(args.path).expanduser().resolve()
    files = _select_files(path, print_all=args.all, num_files=args.num_files)

    for p in files:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            print(f"Skipping non-object JSON: {p}")
            continue
        print_diag(
            path=p,
            payload=payload,
            topk=max(1, int(args.topk)),
            max_context_chars=max(32, int(args.max_context_chars)),
        )


if __name__ == "__main__":
    main()
