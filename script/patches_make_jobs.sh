#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/patches_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model paths differ.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"

TAG_MEANSHIFT="meanshift-r1-distill-qwen7b"

# Remaining 12 seeds after 42-45.
SEEDS_12=(46 47 48 49 50 51 52 53 54 55 56 57)
SEEDS_8=(42 43 44 45 46 47 48 49)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# patches queue jobs
EOF
fi

emit_job() {
  local out="$1"
  local log="$2"
  local out_dir
  out_dir="$(dirname "$out")"
  if [[ "$FILTER_EXISTING" == "1" && -s "$out" ]]; then
    return
  fi
  shift 2
  local -a cmd=("$@")
  local cmd_str=""
  for arg in "${cmd[@]}"; do
    cmd_str+=$(printf "%q " "$arg")
  done
  local out_q log_q
  out_q=$(printf "%q" "$out")
  log_q=$(printf "%q" "$log")
  local out_dir_q
  out_dir_q=$(printf "%q" "$out_dir")
  printf "if [ -s %s ]; then echo \"Skipping existing %s\"; else mkdir -p %s; %s2>&1 | tee %s; fi\n" \
    "$out_q" "$out" "$out_dir_q" "$cmd_str" "$log_q" >> "$JOB_FILE"
}

emit_eval_jobs() {
  local dataset="$1"
  local tag="$2"
  local model="$3"
  local temp="$4"
  local top_p="$5"
  local mode="$6"
  local num_samples="$7"
  shift 7
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.log"
    emit_job "$out" "$log" \
      python utils/llm_eval.py \
      --model_name_or_path "$model" \
      --dataset "$dataset" \
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

# 1) HMMT25: MeanShift for remaining 12 seeds (46-57).
SEEDS=("${SEEDS_12[@]}")
emit_eval_jobs \
  "hmmt25" \
  "$TAG_MEANSHIFT" \
  "$MODEL_BASE" \
  0.798 \
  0.907 \
  "maj@k" \
  16

# 2) GPQA-Diamond: MeanShift for 8 seeds.
SEEDS=("${SEEDS_8[@]}")
emit_eval_jobs \
  "gpqa_diamond" \
  "$TAG_MEANSHIFT" \
  "$MODEL_BASE" \
  0.798 \
  0.907 \
  "maj@k" \
  16

echo "Wrote queue jobs to $JOB_FILE"
