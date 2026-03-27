"""Instruction-following evaluation (IFEval / IFBench).

Generates model responses with vLLM and scores them using the IFEval or IFBench
constraint-checking libraries.  The script follows the same CLI conventions as
``llm_eval.py`` so that AutoDeco / dynamic-sampling flags work out of the box.

Usage examples
--------------
# IFEval (default)
python utils/if_eval.py --model_name_or_path <model> --dataset ifeval

# IFBench
python utils/if_eval.py --model_name_or_path <model> --dataset ifbench

# With AutoDeco heads
python utils/if_eval.py --model_name_or_path <model> --dataset ifeval \
    --autodeco_heads temperature
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm import LLM, SamplingParams
from vllm.autodeco import (
    AUTODECO_HEADS_ARG,
    normalize_autodeco_heads_value,
    validate_autodeco_runtime_extra_args,
)
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Evaluation library imports — lazy so the user only needs deps for the
# benchmark they actually run.
# ---------------------------------------------------------------------------

def _load_eval_lib(dataset: str):
    """Return (evaluation_lib module, read_prompt_list, test_strict, test_loose, print_report)."""
    if dataset == "ifeval":
        from instruction_following_eval import evaluation_lib
    elif dataset == "ifbench":
        from ifbench import evaluation_lib
    else:
        raise ValueError(f"Unknown IF dataset: {dataset}")
    return evaluation_lib


# ---------------------------------------------------------------------------
# Helpers shared with llm_eval.py
# ---------------------------------------------------------------------------

def parse_dynamic_sampling_kv(arg: str):
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Dynamic sampling kwargs must be KEY=VALUE.")
    key, raw_value = arg.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise argparse.ArgumentTypeError("Dynamic sampling kwargs key cannot be empty.")
    raw_value = raw_value.strip()
    try:
        return key, json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            return key, lowered == "true"
        for cast in (int, float):
            try:
                return key, cast(raw_value)
            except ValueError:
                pass
        return key, raw_value


def apply_eval_chat_template(tokenizer, messages, *, reasoning_effort=None):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return tokenizer.apply_chat_template(messages, **kwargs)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning-model outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def write_ascii_table(txt_path: str, dataset_name: str, scores: Dict[str, float]):
    headers = ["", "Score"]
    rows = [[k, f"{v:.4f}"] for k, v in scores.items()]
    col_widths = [
        max(len(headers[0]), *(len(r[0]) for r in rows)) + 2,
        max(len(headers[1]), *(len(r[1]) for r in rows)) + 2,
    ]

    def border():
        return "+" + "+".join("-" * w for w in col_widths) + "+\n"

    lines = [border()]
    lines.append("|" + "|".join(headers[i].center(col_widths[i]) for i in range(2)) + "|\n")
    lines.append(border())
    for r in rows:
        lines.append("|" + "|".join(r[i].center(col_widths[i]) for i in range(2)) + "|\n")
    lines.append(border())
    with open(txt_path, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_model_name(path: str) -> str:
    path = path.rstrip("/")
    parts = path.split("/")
    return parts[-1] if parts else "unknown"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Instruction-following evaluation (IFEval / IFBench) with vLLM.",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ifeval", choices=["ifeval", "ifbench"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the input JSONL. Defaults to data/TempTest/{dataset}.jsonl")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Temperature (default 0 for deterministic IF eval).")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--rp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--strip_think", action="store_true",
                        help="Strip <think>...</think> from responses before scoring.")
    parser.add_argument("--output-file", dest="output_file", type=str, default=None)
    parser.add_argument("--dynamic_sampling_policy", type=str, default=None)
    parser.add_argument("--dynamic_sampling_kwargs", type=str, default="{}")
    parser.add_argument("--dyn", action="append", type=parse_dynamic_sampling_kv, default=[], metavar="KEY=VALUE")
    parser.add_argument("--autodeco_heads", type=str, default=None)
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high"])
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve data path
    # ------------------------------------------------------------------
    data_path = args.data_path or f"data/TempTest/{args.dataset}.jsonl"
    if not os.path.exists(data_path):
        parser.error(f"Data file not found: {data_path}")

    # ------------------------------------------------------------------
    # Load evaluation library
    # ------------------------------------------------------------------
    eval_lib = _load_eval_lib(args.dataset)

    # ------------------------------------------------------------------
    # Read input prompts (native IF schema)
    # ------------------------------------------------------------------
    inputs = eval_lib.read_prompt_list(data_path)
    print(f"Loaded {len(inputs)} prompts from {data_path}")

    # ------------------------------------------------------------------
    # Build extra_args (AutoDeco / dynamic sampling)
    # ------------------------------------------------------------------
    try:
        dynamic_sampling_kwargs = json.loads(args.dynamic_sampling_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"--dynamic_sampling_kwargs must be valid JSON: {exc}")
    if not isinstance(dynamic_sampling_kwargs, dict):
        parser.error("--dynamic_sampling_kwargs must decode to a JSON object.")
    for key, value in args.dyn:
        dynamic_sampling_kwargs[key] = value
    if not dynamic_sampling_kwargs:
        dynamic_sampling_kwargs = None

    autodeco_heads = None
    if args.autodeco_heads is not None:
        try:
            autodeco_heads = normalize_autodeco_heads_value(args.autodeco_heads)
        except ValueError as exc:
            parser.error(f"--autodeco_heads is invalid: {exc}")

    extra_args = None
    if args.dynamic_sampling_policy or autodeco_heads is not None:
        extra_args = {}
    if args.dynamic_sampling_policy:
        extra_args["dynamic_sampling_policy"] = args.dynamic_sampling_policy
        if dynamic_sampling_kwargs is not None:
            extra_args["dynamic_sampling_kwargs"] = dynamic_sampling_kwargs
    elif dynamic_sampling_kwargs is not None:
        parser.error("--dynamic_sampling_kwargs or --dyn requires --dynamic_sampling_policy.")
    if autodeco_heads is not None:
        extra_args[AUTODECO_HEADS_ARG] = autodeco_heads
    if not extra_args:
        extra_args = None

    # ------------------------------------------------------------------
    # vLLM setup
    # ------------------------------------------------------------------
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.seed,
        repetition_penalty=args.rp,
        extra_args=extra_args,
    )

    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp_size, max_model_len=args.max_tokens)
    hf_config = llm.llm_engine.model_config.hf_config
    try:
        validate_autodeco_runtime_extra_args(
            extra_args,
            is_autodeco_model=getattr(hf_config, "model_type", None) == "autodeco",
            enable_temperature_head=getattr(hf_config, "enable_temperature_head", True),
            enable_top_p_head=getattr(hf_config, "enable_top_p_head", True),
        )
    except ValueError as exc:
        parser.error(str(exc))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ------------------------------------------------------------------
    # Build prompts using chat template
    # ------------------------------------------------------------------
    prompts_text: List[str] = []
    for inp in inputs:
        formatted = apply_eval_chat_template(
            tokenizer,
            [{"role": "user", "content": inp.prompt}],
            reasoning_effort=args.reasoning_effort,
        )
        prompts_text.append(formatted)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print(f"Generating responses (temp={args.temp}, max_tokens={args.max_tokens}) ...")
    outputs = llm.generate(prompts_text, sampling_params)

    # ------------------------------------------------------------------
    # Build prompt -> response mapping
    # ------------------------------------------------------------------
    prompt_to_response: Dict[str, str] = {}
    responses: List[str] = []
    for inp, output_group in zip(inputs, outputs):
        response_text = output_group.outputs[0].text
        if args.strip_think:
            response_text = strip_thinking(response_text)
        prompt_to_response[inp.prompt] = response_text
        responses.append(response_text)

    # ------------------------------------------------------------------
    # Evaluate (strict + loose) — collect per-sample results first
    # ------------------------------------------------------------------
    scores: Dict[str, float] = {}
    # per_sample_results[i] = {"strict_follow_all": bool, "strict_follow_instruction_list": [...], ...}
    per_sample_results: List[Dict] = [{} for _ in inputs]

    for func, label in [
        (eval_lib.test_instruction_following_strict, "strict"),
        (eval_lib.test_instruction_following_loose, "loose"),
    ]:
        eval_outputs = [func(inp, prompt_to_response) for inp in inputs]

        for i, o in enumerate(eval_outputs):
            per_sample_results[i][f"{label}_follow_all"] = bool(o.follow_all_instructions)
            per_sample_results[i][f"{label}_follow_instruction_list"] = [
                bool(b) for b in o.follow_instruction_list
            ]

        prompt_correct = sum(1 for o in eval_outputs if o.follow_all_instructions)
        prompt_total = len(eval_outputs)
        instr_correct = sum(sum(o.follow_instruction_list) for o in eval_outputs)
        instr_total = sum(len(o.follow_instruction_list) for o in eval_outputs)

        prompt_acc = prompt_correct / prompt_total
        instr_acc = instr_correct / instr_total

        scores[f"prompt_{label}"] = prompt_acc
        scores[f"instruction_{label}"] = instr_acc

        print(f"\n{'=' * 64}")
        print(f"{label.upper()} results:")
        print(f"  prompt-level:      {prompt_acc:.4f} ({prompt_correct}/{prompt_total})")
        print(f"  instruction-level: {instr_acc:.4f} ({instr_correct}/{instr_total})")

    # ------------------------------------------------------------------
    # Build log path
    # ------------------------------------------------------------------
    ckpt_name = extract_model_name(args.model_name_or_path)
    dyn_tag = f"-dyn_{args.dynamic_sampling_policy}" if args.dynamic_sampling_policy else ""
    autodeco_heads_tag = ""
    if autodeco_heads is not None:
        head_label = "none" if not autodeco_heads else "_".join(autodeco_heads)
        autodeco_heads_tag = f"-autodeco_heads_{head_label}"
    reasoning_effort_tag = f"-reasoning_effort_{args.reasoning_effort}" if args.reasoning_effort else ""
    strip_tag = "-strip_think" if args.strip_think else ""

    log_dir = f"generation_log/{args.dataset}"
    os.makedirs(log_dir, exist_ok=True)
    log_base = (
        f"{log_dir}/{ckpt_name}-temp{args.temp}-top_p{args.top_p}"
        f"-top_k{args.top_k}-rp{args.rp}-max_tokens{args.max_tokens}-seed{args.seed}"
        f"{reasoning_effort_tag}{dyn_tag}{autodeco_heads_tag}{strip_tag}"
    )

    # ------------------------------------------------------------------
    # Save rich JSONL (response + per-instruction eval results)
    # ------------------------------------------------------------------
    if args.output_file:
        save_dir = os.path.dirname(args.output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            for inp, response_text, res in zip(inputs, responses, per_sample_results):
                record = {
                    "prompt": inp.prompt,
                    "response": response_text,
                    "metadata": {
                        "dataset": args.dataset,
                        "key": getattr(inp, "key", None),
                        "instruction_id_list": list(getattr(inp, "instruction_id_list", [])),
                        "model_name_or_path": args.model_name_or_path,
                        "ckpt_name": ckpt_name,
                        "temp": args.temp,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "rp": args.rp,
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                        "reasoning_effort": args.reasoning_effort,
                        "dynamic_sampling_policy": args.dynamic_sampling_policy,
                        "dynamic_sampling_kwargs": dynamic_sampling_kwargs,
                        "autodeco_heads": autodeco_heads,
                        "strip_think": args.strip_think,
                        **res,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(inputs)} records -> {args.output_file}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    avg = sum(scores.values()) / len(scores)
    print(f"Average (all 4 metrics): {avg:.4f}")
    scores["average"] = avg

    txt_path = f"{log_base}.txt"
    write_ascii_table(txt_path, args.dataset, scores)
    print(f"Summary table -> {txt_path}")


if __name__ == "__main__":
    main()
