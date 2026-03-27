from vllm import LLM, SamplingParams
from vllm.autodeco import (
    AUTODECO_HEADS_ARG,
    normalize_autodeco_heads_value,
    validate_autodeco_runtime_extra_args,
)
import json
from boxed_extract import *
import argparse
import os
from collections import OrderedDict
from boxed_extract import *
from transformers import AutoTokenizer
import math
# from model.templlm_auto import AutoDecoModelForCausalLM


def write_ascii_table(txt_path: str, dataset_name: str, avg_acc: float):
    headers = ["", "Acc"]
    row = [dataset_name, f"{avg_acc:.2f}"]
    col_widths = [max(len(headers[i]), len(row[i])) + 2 for i in range(2)]

    def make_border() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+\n"

    border = make_border()
    header_line = "|" + "|".join(headers[i].center(col_widths[i]) for i in range(2)) + "|\n"
    data_line = "|" + "|".join(row[i].center(col_widths[i]) for i in range(2)) + "|\n"

    table_str = border + header_line + border + data_line + border
    with open(txt_path, "w") as txt_file:
        txt_file.write(table_str)


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


DEFAULT_FINAL_ANSWER_SUFFIX = "Make sure you output the final answer within \\boxed{}."
MCQ_FINAL_ANSWER_SUFFIX = (
    # "Make sure you output the final answer within \\boxed{}."
    # ===================================================================================================
    #   We found that this prompt below actually works better, but to keep evaluation fair and to avoid
    #   giving the model any hints, we do the version on line 66 for our main results.
    # ===================================================================================================
    "Put your final letter answer within \\boxed{}, for example \\boxed{A}. Exactly one answer choice is correct."
)


def get_final_answer_suffix(ground_truth) -> str:
    if normalize_multiple_choice_answer(str(ground_truth)) is not None:
        return MCQ_FINAL_ANSWER_SUFFIX
    return DEFAULT_FINAL_ANSWER_SUFFIX


def build_problem_prompt(problem: str, ground_truth) -> str:
    return f"{problem.rstrip()}\n{get_final_answer_suffix(ground_truth)}"


if __name__ == "__main__":
    def extract_model_name(path):
        path = path.rstrip('/')
        parts = path.split('/')
        if len(parts) >= 2:
            return parts[-1]
        elif len(parts) == 1:
            return parts[0]
        else:
            return 'unknown'


    dynamic_policy_help = (
        "Dynamic sampling policies and kwargs:\n"
        "  confidence_gated: T_high (>=0), maxprob_threshold (0..1)\n"
        "  edt: T0 (>=0), theta (>=0), N (0..1)\n"
        "  entropy_continuous: T_min (>=0), T_max (>=0 and >=T_min)\n"
        "  entropy_shift: T_base (>=0), delta (>=0), H_mean (0..1)\n"
        "  entropy_adaptive: H_threshold (0..1, normalized), T_low (>=0), T_high (>=0)\n"
        "Example:\n"
        "  --dynamic_sampling_policy edt "
        "--dyn T0=0.798 --dyn theta=0.1 --dyn N=0.8"
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dynamic_policy_help,
    )
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--rp', type=float, default=1.0)
    parser.add_argument('--num_samples', '--k', type=int, default=16)
    parser.add_argument('--mode', type=str, default='maj@k', choices=['pass@k', 'maj@k'])
    parser.add_argument('--model_name_or_path', type=str, default='/apdcephfs_qy3/share_301812049/shared/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='aime24')
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--output-file', dest='output_file', type=str, default=None,
                        help='Optional jsonl output path for per-sample generations.')
    parser.add_argument('--dynamic_sampling_policy', type=str, default=None,
                        help='Optional dynamic sampling policy '
                             '(confidence_gated, edt, entropy_continuous, entropy_shift, entropy_adaptive).')
    parser.add_argument('--dynamic_sampling_kwargs', type=str, default='{}',
                        help='Optional JSON object of kwargs for dynamic sampling policy. '
                             'Overrides can be provided via --dyn.')
    parser.add_argument('--dyn', action='append', type=parse_dynamic_sampling_kv, default=[],
                        metavar='KEY=VALUE',
                        help='Repeatable override for dynamic sampling kwargs, e.g. --dyn window=16 --dyn alpha=0.5')
    parser.add_argument(
        '--autodeco_heads',
        type=str,
        default=None,
        help='Optional comma-separated AutoDeco heads to enable '
             '(temperature, top_p, both, none). Omit to use all available heads.',
    )
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=['low', 'medium', 'high'],
                        help='Optional GPT-OSS reasoning effort used when rendering the chat template.')
    args = parser.parse_args()


    ckpt_name = extract_model_name(args.model_name_or_path)
    temp = args.temp
    k = args.num_samples
    seed = args.seed

    with open(f'data/TempTest/{args.dataset}.jsonl', 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

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

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=k,
        seed=seed,
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

    tokenizer = llm.get_tokenizer()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    problems = [
        apply_eval_chat_template(
            tokenizer,
            [{"role": "user", "content": build_problem_prompt(item['problem'], item['gt'])}],
            reasoning_effort=args.reasoning_effort,
        ) for item in data
    ]

    ground_truths = [
        item['gt'] for item in data
    ]

    if not os.path.exists(f'generation_log/{args.dataset}'):
        os.makedirs(f'generation_log/{args.dataset}')
    outputs = llm.generate(problems, sampling_params)
    save_outputs_f = None
    if args.output_file:
        save_dir = os.path.dirname(args.output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_outputs_f = open(args.output_file, 'w')

    def aggregate_score(scores, mode):
        if mode == 'pass@k':
            return 1.0 if max(scores) > 0.5 else 0.0
        # maj@k
        return 1.0 if (sum(scores) / len(scores)) > 0.5 else 0.0

    dyn_tag = f"-dyn_{args.dynamic_sampling_policy}" if args.dynamic_sampling_policy else ""
    autodeco_heads_tag = ""
    if autodeco_heads is not None:
        head_label = "none" if not autodeco_heads else "_".join(autodeco_heads)
        autodeco_heads_tag = f"-autodeco_heads_{head_label}"
    reasoning_effort_tag = (
        f"-reasoning_effort_{args.reasoning_effort}" if args.reasoning_effort else ""
    )
    log_base = (
        f'generation_log/{args.dataset}/{ckpt_name}-temp{temp}-top_p{args.top_p}'
        f'-top_k{args.top_k}-rp{args.rp}-max_tokens{args.max_tokens}-seed{seed}'
        f'{reasoning_effort_tag}{dyn_tag}{autodeco_heads_tag}'
    )
    all_acc = []
    for idx, output_group in enumerate(outputs):
        gt = str(ground_truths[idx])
        scores = []
        for sample_idx, output in enumerate(output_group.outputs):
            generated_text = output.text
            temp = getattr(output, 'temperatures', None)  # 使用当前循环的temp值作为默认值
            top_p = getattr(output, 'top_ps', None)
            score = compute_score(generated_text, gt)
            scores.append(score)
            if save_outputs_f is not None:
                save_outputs_f.write(json.dumps({
                    'prompt': problems[idx],
                    'response': generated_text,
                    'metadata': {
                        'dataset': args.dataset,
                        'problem_index': idx,
                        'sample_index': sample_idx,
                        'ground_truth': ground_truths[idx],
                        'score': score,
                        'temp': temp if temp is not None else args.temp,
                        'top_p': top_p if top_p is not None else args.top_p,
                        'top_k': args.top_k,
                        'rp': args.rp,
                        'max_tokens': args.max_tokens,
                        'mode': args.mode,
                        'seed': args.seed,
                        'model_name_or_path': args.model_name_or_path,
                        'ckpt_name': ckpt_name,
                        'reasoning_effort': args.reasoning_effort,
                        'dynamic_sampling_policy': args.dynamic_sampling_policy,
                        'dynamic_sampling_kwargs': dynamic_sampling_kwargs,
                        'autodeco_heads': autodeco_heads,
                    }
                }, ensure_ascii=False) + '\n')
        problem_acc = round(aggregate_score(scores, args.mode) * 100, 2)
        all_acc.append(problem_acc)

    avg_acc = round(sum(all_acc)/len(all_acc), 2)
    print(f"Overall avg Acc: {avg_acc}%")

    txt_path = f"{log_base}.txt"
    write_ascii_table(txt_path, args.dataset, avg_acc)
    if save_outputs_f is not None:
        save_outputs_f.close()
