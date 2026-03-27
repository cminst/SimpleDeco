#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="aime24"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/aime24_factorization_ablation_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model paths differ.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-R1-Distill-Qwen-7B-merged}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-0.798}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.907}"

TAG_AUTODECO="${TAG_AUTODECO:-autodeco-r1-distill-qwen7b}"
TAG_MEANSHIFT="${TAG_MEANSHIFT:-meanshift-r1-distill-qwen7b}"
TAG_TEMP_HEAD_ONLY="${TAG_TEMP_HEAD_ONLY:-temphead-meanshift-topp-r1-distill-qwen7b}"
TAG_TOP_P_HEAD_ONLY="${TAG_TOP_P_HEAD_ONLY:-topphead-meanshift-temp-r1-distill-qwen7b}"

SEEDS_12=(42 43 44 45 46 47 48 49 50 51 52 53)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# aime24 factorization ablation queue jobs
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
    local out="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN"
      utils/llm_eval.py
      --model_name_or_path "$model"
      --dataset "$DATASET"
      --temp "$temp"
      --top_p "$top_p"
      --mode "maj@k"
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

SEEDS=("${SEEDS_12[@]}")

# 1) AutoDeco full controller.
emit_eval_jobs "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 1.0 "$NUM_SAMPLES"

# 2) MeanShift baseline.
emit_eval_jobs "$TAG_MEANSHIFT" "$MODEL_BASE" "$MEANSHIFT_TEMP" "$MEANSHIFT_TOP_P" "$NUM_SAMPLES"

# 3) Temperature head only, with MeanShift top-p.
emit_eval_jobs \
  "$TAG_TEMP_HEAD_ONLY" \
  "$MODEL_AUTODECO" \
  1.0 \
  "$MEANSHIFT_TOP_P" \
  "$NUM_SAMPLES" \
  --autodeco_heads temperature

# 4) Top-p head only, with MeanShift temperature.
emit_eval_jobs \
  "$TAG_TOP_P_HEAD_ONLY" \
  "$MODEL_AUTODECO" \
  "$MEANSHIFT_TEMP" \
  1.0 \
  "$NUM_SAMPLES" \
  --autodeco_heads top_p

echo "Wrote queue jobs to $JOB_FILE"
