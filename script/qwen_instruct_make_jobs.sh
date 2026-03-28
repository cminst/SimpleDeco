#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/qwen_instruct_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"
RESULT_ROOT="${RESULT_ROOT:-ckpt_qwen_instruct}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-Qwen3-30B-Instruct-Merged}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-maj@k}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

BASE_TEMP="${BASE_TEMP:-0.7}"
BASE_TOP_P="${BASE_TOP_P:-0.8}"
AUTODECO_TEMP="${AUTODECO_TEMP:-1.0}"
AUTODECO_TOP_P="${AUTODECO_TOP_P:-1.0}"
MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-0.861}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.843}"
GREEDY_TEMP="${GREEDY_TEMP:-0.0}"
GREEDY_TOP_P="${GREEDY_TOP_P:-0.95}"

TAG_BASE="${TAG_BASE:-base-qwen3-30b-instruct}"
TAG_AUTODECO="${TAG_AUTODECO:-autodeco-qwen3-30b-instruct}"
TAG_MEANSHIFT="${TAG_MEANSHIFT:-meanshift-qwen3-30b-instruct}"
TAG_GREEDY="${TAG_GREEDY:-greedy-qwen3-30b-instruct}"

DATASETS=(aime24 hmmt25 gpqa_diamond mmlu_pro_lite)
SEEDS_8=(42 43 44 45 46 47 48 49)
SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# qwen instruct queue jobs
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

set_dataset_seeds() {
  local dataset="$1"
  case "$dataset" in
    aime24|gpqa_diamond|mmlu_pro_lite)
      SEEDS=("${SEEDS_8[@]}")
      ;;
    hmmt25)
      SEEDS=("${SEEDS_16[@]}")
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

emit_eval_jobs() {
  local dataset="$1"
  local tag="$2"
  local temp="$3"
  local top_p="$4"
  local num_samples="$5"
  shift 5
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="${RESULT_ROOT}/${dataset}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="${RESULT_ROOT}/${dataset}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN"
      utils/llm_eval.py
      --model_name_or_path "$MODEL_AUTODECO"
      --dataset "$dataset"
      --temp "$temp"
      --top_p "$top_p"
      --mode "$MODE"
      --num_samples "$num_samples"
      --tp_size "$TP_SIZE"
      --max_tokens "$MAX_TOKENS"
      --seed "$seed"
      --output-file "$out"
    )

    if ((${#extra_args[@]} > 0)); then
      cmd+=("${extra_args[@]}")
    fi

    emit_job "$out" "$log" "${cmd[@]}"
  done
}

for dataset in "${DATASETS[@]}"; do
  set_dataset_seeds "$dataset"

  emit_eval_jobs \
    "$dataset" \
    "$TAG_BASE" \
    "$BASE_TEMP" \
    "$BASE_TOP_P" \
    "$NUM_SAMPLES" \
    --autodeco_heads none

  emit_eval_jobs \
    "$dataset" \
    "$TAG_AUTODECO" \
    "$AUTODECO_TEMP" \
    "$AUTODECO_TOP_P" \
    "$NUM_SAMPLES" \
    --autodeco_heads temperature top_p

  emit_eval_jobs \
    "$dataset" \
    "$TAG_MEANSHIFT" \
    "$MEANSHIFT_TEMP" \
    "$MEANSHIFT_TOP_P" \
    "$NUM_SAMPLES" \
    --autodeco_heads none

  SEEDS=(42)
  emit_eval_jobs \
    "$dataset" \
    "$TAG_GREEDY" \
    "$GREEDY_TEMP" \
    "$GREEDY_TOP_P" \
    1 \
    --autodeco_heads none
done

echo "Wrote queue jobs to $JOB_FILE"
