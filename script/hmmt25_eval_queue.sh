#!/usr/bin/env bash
set -euo pipefail

# Queue for hmmt25 evals (fills missing seeds, then extends to 16 seeds).
# Skips any output JSONL that already exists.

DATASET="hmmt25"
GPU_ID="${GPU_ID:-0}"

# Update these if your model paths differ.
MODEL_BASE="ckpt/DeepSeek-R1-Distill-Qwen-7B"
MODEL_AUTODECO="ckpt/AutoDeco-R1-Distill-Qwen-7B-merged"

TAG_BASE="base-r1-distill-qwen7b"
TAG_AUTODECO="autodeco-r1-distill-qwen7b"
TAG_CONFGATE="confgate-0.6-0.9-r1-distill-qwen7b"
TAG_MEANSHIFT="meanshift-0.72-0.79-r1-distill-qwen7b"
TAG_GREEDY="greedy-r1-distill-qwen7b"

SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)
SEEDS_EXTRA_8=(50 51 52 53 54 55 56 57)

run_eval () {
  local tag="$1"
  local model="$2"
  local temp="$3"
  local top_p="$4"
  local mode="$5"
  local num_samples="$6"
  shift 6
  local extra_args=("$@")

  mkdir -p "ckpt/${DATASET}/${tag}"

  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export VLLM_DISABLE_COMPILE_CACHE=1

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.log"
    if [[ -s "$out" ]]; then
      echo "Skipping existing $out"
      continue
    fi
    python utils/llm_eval.py \
      --model_name_or_path "$model" \
      --dataset "$DATASET" \
      --temp "$temp" \
      --top_p "$top_p" \
      --mode "$mode" \
      --num_samples "$num_samples" \
      --tp_size 1 \
      --max_tokens 32768 \
      --seed "$seed" \
      --save_outputs "$out" \
      "${extra_args[@]}" \
      2>&1 | tee "$log"
  done
}

# 1) Confgate 0.6-0.9: fill holes (43/45/47/49) and add 50-57.
SEEDS=("${SEEDS_16[@]}")
run_eval \
  "$TAG_CONFGATE" \
  "$MODEL_BASE" \
  1.0 \
  0.95 \
  "pass@k" \
  16 \
  --dynamic_sampling_policy confidence_gated \
  --dynamic_sampling_kwargs '{"maxprob_threshold": 0.6, "T_high": 0.9}'

# 2) Base + AutoDeco: add 8 more seeds (50-57).
SEEDS=("${SEEDS_EXTRA_8[@]}")
run_eval "$TAG_BASE" "$MODEL_BASE" 0.6 0.95 "maj@k" 16
run_eval "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 1.0 "maj@k" 16

# 3) Meanshift: 16 seeds with new temp/top-p.
SEEDS=("${SEEDS_16[@]}")
run_eval "$TAG_MEANSHIFT" "$MODEL_BASE" 0.72 0.79 "maj@k" 16

# 4) Greedy: one seed, one sample.
SEEDS=(42)
run_eval "$TAG_GREEDY" "$MODEL_BASE" 0.0 0.95 "maj@k" 1
