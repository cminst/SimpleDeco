#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="mmlu_pro_lite"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/mmlu_pro_lite_gptoss20b_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"
RESULT_ROOT="${RESULT_ROOT:-ckpt_gptoss20b}"
MODEL_BASE="${MODEL_BASE:-ckpt/gpt-oss-20b}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-GPT-OSS-20B-Merged}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-maj@k}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-1.032}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.940}"

case "$REASONING_EFFORT" in
  ""|low|medium|high) ;;
  *)
    echo "Invalid REASONING_EFFORT: '$REASONING_EFFORT'. Expected one of: low, medium, high." >&2
    exit 1
    ;;
esac

REASONING_TAG_SUFFIX=""
if [[ -n "$REASONING_EFFORT" ]]; then
  REASONING_TAG_SUFFIX="-effort-${REASONING_EFFORT}"
fi

TAG_BASE="base-gptoss20b${REASONING_TAG_SUFFIX}"
TAG_AUTODECO="autodeco-gptoss20b${REASONING_TAG_SUFFIX}"
TAG_MEANSHIFT="meanshift-${MEANSHIFT_TEMP}-${MEANSHIFT_TOP_P}-gptoss20b${REASONING_TAG_SUFFIX}"
TAG_GREEDY="greedy-gptoss20b${REASONING_TAG_SUFFIX}"
SEEDS_8=(42 43 44 45 46 47 48 49)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# mmlu_pro_lite GPT-OSS-20B queue jobs
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

emit_eval_jobs() {
  local tag="$1"
  local model="$2"
  local temp="$3"
  local top_p="$4"
  local num_samples="$5"
  shift 5
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="${RESULT_ROOT}/${DATASET}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="${RESULT_ROOT}/${DATASET}/${tag}/maj${num_samples}_seed${seed}.log"

    local -a cmd=(
      "$PYTHON_BIN"
      utils/llm_eval.py
      --model_name_or_path "$model"
      --dataset "$DATASET"
      --temp "$temp"
      --top_p "$top_p"
      --mode "$MODE"
      --num_samples "$num_samples"
      --tp_size "$TP_SIZE"
      --max_tokens "$MAX_TOKENS"
      --seed "$seed"
      --output-file "$out"
    )

    if [[ -n "$REASONING_EFFORT" ]]; then
      cmd+=(--reasoning_effort "$REASONING_EFFORT")
    fi

    if ((${#extra_args[@]} > 0)); then
      cmd+=("${extra_args[@]}")
    fi

    emit_job "$out" "$log" "${cmd[@]}"
  done
}

SEEDS=("${SEEDS_8[@]}")

# 1) Base: all 8 seeds.
emit_eval_jobs "$TAG_BASE" "$MODEL_BASE" 1.0 1.0 "$NUM_SAMPLES"

# 2) AutoDeco: all 8 seeds.
emit_eval_jobs "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 0.95 "$NUM_SAMPLES"

# 3) Meanshift: all 8 seeds with adjusted temp/top-p.
emit_eval_jobs "$TAG_MEANSHIFT" "$MODEL_BASE" "$MEANSHIFT_TEMP" "$MEANSHIFT_TOP_P" "$NUM_SAMPLES"

# 4) Greedy: one seed, one sample.
SEEDS=(42)
emit_eval_jobs "$TAG_GREEDY" "$MODEL_BASE" 0.0 0.95 1

echo "Wrote queue jobs to $JOB_FILE"
