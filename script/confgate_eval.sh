#!/usr/bin/env bash
set -euo pipefail

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATASET=hmmt25
TOP_P=0.95

run_job () {
  local gpu="$1"
  local tag="$2"
  local kwargs="$3"
  shift 3
  local seeds=("$@")

  mkdir -p "ckpt/${DATASET}/${tag}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_DISABLE_COMPILE_CACHE=1
    for seed in "${seeds[@]}"; do
      python utils/llm_eval.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --temp 1.0 \
        --top_p "$TOP_P" \
        --mode pass@k \
        --num_samples 16 \
        --tp_size 1 \
        --max_tokens 32768 \
        --dynamic_sampling_policy confidence_gated \
        --dynamic_sampling_kwargs "$kwargs" \
        --seed "$seed" \
        --save_outputs "ckpt/${DATASET}/${tag}/maj16_seed${seed}.jsonl" \
        2>&1 | tee "ckpt/${DATASET}/${tag}/maj16_seed${seed}.log"
    done
  ) &
}

run_job 0 "confgate-0.7-0.9-r1-distill-qwen7b" '{"maxprob_threshold": 0.7, "T_high": 0.9}' 42 43
run_job 1 "confgate-0.7-0.9-r1-distill-qwen7b" '{"maxprob_threshold": 0.7, "T_high": 0.9}' 44 45
run_job 2 "confgate-0.7-0.9-r1-distill-qwen7b" '{"maxprob_threshold": 0.7, "T_high": 0.9}' 46 47
run_job 3 "confgate-0.7-0.9-r1-distill-qwen7b" '{"maxprob_threshold": 0.7, "T_high": 0.9}' 48 49

wait

MODEL2=cminst/AutoDeco-R1-Distill-Qwen-7B-merged
TAG2="autodeco-r1-distill-qwen7b"

run_job_autodeco () {
  local gpu="$1"
  local tag="$2"
  shift 2
  local seeds=("$@")

  mkdir -p "ckpt/${DATASET}/${tag}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_DISABLE_COMPILE_CACHE=1
    for seed in "${seeds[@]}"; do
      python utils/llm_eval.py \
        --model_name_or_path "$MODEL2" \
        --dataset "$DATASET" \
        --temp 1.0 \
        --top_p 1.0 \
        --mode maj@k \
        --num_samples 16 \
        --tp_size 1 \
        --max_tokens 32768 \
        --seed "$seed" \
        --save_outputs "ckpt/${DATASET}/${tag}/maj16_seed${seed}.jsonl" \
        2>&1 | tee "ckpt/${DATASET}/${tag}/maj16_seed${seed}.log"
    done
  ) &
}

run_job_autodeco 0 "$TAG2" 42 43
run_job_autodeco 1 "$TAG2" 44 45
run_job_autodeco 2 "$TAG2" 46 47
run_job_autodeco 3 "$TAG2" 48 49

wait
