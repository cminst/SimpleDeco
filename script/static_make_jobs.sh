#!/usr/bin/env bash
# Generates static baseline jobs (t=0.8, top_p=0.95) on the base model
# across all four main benchmarks:
#   aime24        — 8 seeds  (42-49)
#   hmmt25        — 16 seeds (42-57)
#   gpqa_diamond  — 8 seeds  (42-49)
#   mmlu_pro_lite — 8 seeds  (42-49)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/static_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"

TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.95}"
TAG="${TAG:-static-t${TEMP}-p${TOP_P}}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

SEEDS_8=(42 43 44 45 46 47 48 49)
SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# static baseline (t=${TEMP}, top_p=${TOP_P}) jobs
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
  local out_q log_q out_dir_q
  out_q=$(printf "%q" "$out")
  log_q=$(printf "%q" "$log")
  out_dir_q=$(printf "%q" "$out_dir")
  printf "if [ -s %s ]; then echo \"Skipping existing %s\"; else mkdir -p %s; %s2>&1 | tee %s; fi\n" \
    "$out_q" "$out" "$out_dir_q" "$cmd_str" "$log_q" >> "$JOB_FILE"
}

emit_dataset_jobs() {
  local dataset="$1"
  shift
  local -a seeds=("$@")

  for seed in "${seeds[@]}"; do
    local out="ckpt/${dataset}/${TAG}/maj${NUM_SAMPLES}_seed${seed}.jsonl"
    local log="ckpt/${dataset}/${TAG}/maj${NUM_SAMPLES}_seed${seed}.log"
    emit_job "$out" "$log" \
      python utils/llm_eval.py \
      --model_name_or_path "$MODEL_BASE" \
      --dataset "$dataset" \
      --temp "$TEMP" \
      --top_p "$TOP_P" \
      --mode "maj@k" \
      --num_samples "$NUM_SAMPLES" \
      --tp_size "$TP_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --seed "$seed" \
      --output-file "$out"
  done
}

emit_dataset_jobs aime24        "${SEEDS_8[@]}"
emit_dataset_jobs hmmt25        "${SEEDS_16[@]}"
emit_dataset_jobs gpqa_diamond  "${SEEDS_8[@]}"
emit_dataset_jobs mmlu_pro_lite "${SEEDS_8[@]}"

echo "Wrote queue jobs to $JOB_FILE"
