#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="hmmt25"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/hmmt25_jobs.txt}"
APPEND="${APPEND:-0}"

# Update if your model paths differ.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-R1-Distill-Qwen-7B-merged}"

TAG_BASE="base-r1-distill-qwen7b"
TAG_AUTODECO="autodeco-r1-distill-qwen7b"
TAG_CONFGATE="confgate-0.6-0.9-r1-distill-qwen7b"
TAG_MEANSHIFT="meanshift-0.72-0.79-r1-distill-qwen7b"
TAG_GREEDY="greedy-r1-distill-qwen7b"

SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)
SEEDS_EXTRA_8=(50 51 52 53 54 55 56 57)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# hmmt25 queue jobs
EOF
fi

emit_job() {
  local out="$1"
  local log="$2"
  local out_dir
  out_dir="$(dirname "$out")"
  shift 2
  local -a cmd=("$@")
  local cmd_str=""
  for arg in "${cmd[@]}"; do
    cmd_str+=$(printf "%q " "$arg")
  done
  local out_q log_q
  out_q=$(printf "%q" "$out")
  log_q=$(printf "%q" "$log")
  local mkdir_cmd
  mkdir_cmd=$(printf "%q" "mkdir -p $out_dir")
  printf "if [ -s %s ]; then echo \"Skipping existing %s\"; else %s; %s2>&1 | tee %s; fi\n" \
    "$out_q" "$out" "$mkdir_cmd" "$cmd_str" "$log_q" >> "$JOB_FILE"
}

emit_eval_jobs() {
  local tag="$1"
  local model="$2"
  local temp="$3"
  local top_p="$4"
  local mode="$5"
  local num_samples="$6"
  shift 6
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.log"
    emit_job "$out" "$log" \
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
      "${extra_args[@]}"
  done
}

# 1) Confgate 0.6-0.9: fill holes (43/45/47/49) and add 50-57.
SEEDS=("${SEEDS_16[@]}")
emit_eval_jobs \
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
emit_eval_jobs "$TAG_BASE" "$MODEL_BASE" 0.6 0.95 "maj@k" 16
emit_eval_jobs "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 1.0 "maj@k" 16

# 3) Meanshift: 16 seeds with new temp/top-p.
SEEDS=("${SEEDS_16[@]}")
emit_eval_jobs "$TAG_MEANSHIFT" "$MODEL_BASE" 0.72 0.79 "maj@k" 16

# 4) Greedy: one seed, one sample.
SEEDS=(42)
emit_eval_jobs "$TAG_GREEDY" "$MODEL_BASE" 0.0 0.95 "maj@k" 1

echo "Wrote queue jobs to $JOB_FILE"
