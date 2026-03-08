#!/usr/bin/env python3
"""
Post-hoc metric calculator for LLM evaluation results.

This script calculates pass@k or maj@k metrics from existing JSONL evaluation results
without needing to re-run the full evaluation.

Usage:
    python calc_post_hoc.py --input ckpt/qwen3_4b_thinking_aime24_maj8_seed42.jsonl --config pass8
    python calc_post_hoc.py --input ckpt/simpledeco_ql_aime24_maj8_seed42.jsonl --config pass4
    python calc_post_hoc.py --input ckpt/*.jsonl --config pass1

Configs:
    pass<k>: pass if any of the first k samples is correct
    maj<k>: pass if majority (>k/2) of the first k samples are correct

Note: k cannot exceed the number of samples available in the input files.
"""

import argparse
import json
from collections import defaultdict
import os
import glob


def parse_config(config):
    """Parse config string like 'pass8', 'maj4', etc."""
    if config.startswith("pass"):
        mode = "pass"
        k = int(config[4:])
    elif config.startswith("maj"):
        mode = "maj"
        k = int(config[3:])
    else:
        raise ValueError(f"Invalid config: {config}. Must be like 'pass8' or 'maj4'")
    return mode, k


def load_jsonl_files(file_pattern):
    """Load data from JSONL file(s)."""
    files = glob.glob(file_pattern) if "*" in file_pattern else [file_pattern]

    all_data = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: File not found: {f}")
            continue
        with open(f) as fp:
            for line in fp:
                all_data.append(json.loads(line))

    return all_data


def aggregate_score(scores, mode):
    """Aggregate scores based on mode."""
    if mode == "pass":
        return 1.0 if max(scores) > 0.5 else 0.0
    elif mode == "maj":
        return 1.0 if (sum(scores) / len(scores)) > 0.5 else 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_metric(data, mode, k):
    """Calculate the metric for given mode and k."""
    # Group by problem_index
    problems = defaultdict(list)
    for d in data:
        problems[d["metadata"]["problem_index"]].append(d)

    # Sort problems by index
    problem_indices = sorted(problems.keys())

    # Check if we have enough samples
    num_samples_available = len(problems[problem_indices[0]]) if problem_indices else 0
    if k > num_samples_available:
        raise ValueError(
            f"Cannot calculate {mode}{k}: only {num_samples_available} samples available per problem. "
            f"Maximum k is {num_samples_available}."
        )

    all_acc = []

    for idx in problem_indices:
        samples = problems[idx]
        # Take first k samples
        samples_k = samples[:k]
        scores = [s["metadata"]["score"] for s in samples_k]

        problem_acc = aggregate_score(scores, mode)
        all_acc.append(problem_acc)

    avg_acc = sum(all_acc) / len(all_acc) if all_acc else 0
    return avg_acc * 100, len(all_acc)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pass@k or maj@k from existing evaluation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file(s), supports glob patterns like ckpt/*.jsonl",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config like pass8, pass4, pass1, maj8, maj4, etc.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-problem results"
    )

    args = parser.parse_args()

    mode, k = parse_config(args.config)

    print(f"Loading data from: {args.input}")
    data = load_jsonl_files(args.input)

    if not data:
        print("Error: No data loaded")
        return

    # Group by problem to count
    problems = defaultdict(list)
    for d in data:
        problems[d["metadata"]["problem_index"]].append(d)

    num_problems = len(problems)
    num_samples = len(list(problems.values())[0]) if problems else 0

    print(
        f"Loaded {len(data)} samples across {num_problems} problems ({num_samples} samples/problem)"
    )
    print(f"Calculating {mode}@{k}...")

    try:
        avg_acc, count = calculate_metric(data, mode, k)
        print(f"\n{'=' * 40}")
        print(f"Result: {mode}@{k} = {avg_acc:.2f}% ({count} problems)")
        print(f"{'=' * 40}")

        if args.verbose:
            print("\nPer-problem results:")
            print("-" * 30)
            problem_indices = sorted(problems.keys())
            for idx in problem_indices:
                samples = problems[idx][:k]
                scores = [s["metadata"]["score"] for s in samples]
                result = aggregate_score(scores, mode)
                gt = samples[0]["metadata"]["ground_truth"]
                print(
                    f"Problem {idx:2d}: scores={scores} -> {'PASS' if result else 'FAIL'} (gt={gt})"
                )

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
