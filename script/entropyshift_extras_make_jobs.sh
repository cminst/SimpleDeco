#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/entropyshift_extras_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model path differs.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
ENTROPY_SHIFT_DELTA="${ENTROPY_SHIFT_DELTA:-0.10}"
ENTROPY_MEAN="${ENTROPY_MEAN:-0.07197317484105381}"
TAG_ENTROPYSHIFT="${TAG_ENTROPYSHIFT:-entropyshift-0.10-r1-distill-qwen7b}"

HMMT25_TEMP="${HMMT25_TEMP:-0.72}"
HMMT25_TOP_P="${HMMT25_TOP_P:-0.79}"
GPQA_TEMP="${GPQA_TEMP:-0.72}"
GPQA_TOP_P="${GPQA_TOP_P:-0.79}"

SEEDS_HMMT25=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)
SEEDS_GPQA=(42 43 44 45 46 47 48 49)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# entropyshift extras queue jobs
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
  local temp="$3"
  local top_p="$4"
  local mode="$5"
  local num_samples="$6"
  shift 6
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN" utils/llm_eval.py
      --model_name_or_path "$MODEL_BASE"
      --dataset "$dataset"
      --temp "$temp"
      --top_p "$top_p"
      --mode "$mode"
      --num_samples "$num_samples"
      --tp_size "$TP_SIZE"
      --max_tokens "$MAX_TOKENS"
      --seed "$seed"
      --save_outputs "$out"
    )
    if ((${#extra_args[@]} > 0)); then
      cmd+=("${extra_args[@]}")
    fi
    emit_job "$out" "$log" "${cmd[@]}"
  done
}

dyn_kwargs=$(printf '{"T_base": %s, "delta": %s, "H_mean": %s}' \
  "$HMMT25_TEMP" "$ENTROPY_SHIFT_DELTA" "$ENTROPY_MEAN")
SEEDS=("${SEEDS_HMMT25[@]}")
emit_eval_jobs \
  "hmmt25" \
  "$TAG_ENTROPYSHIFT" \
  "$HMMT25_TEMP" \
  "$HMMT25_TOP_P" \
  "maj@k" \
  "$NUM_SAMPLES" \
  --dynamic_sampling_policy entropy_shift \
  --dynamic_sampling_kwargs "$dyn_kwargs"

dyn_kwargs=$(printf '{"T_base": %s, "delta": %s, "H_mean": %s}' \
  "$GPQA_TEMP" "$ENTROPY_SHIFT_DELTA" "$ENTROPY_MEAN")
SEEDS=("${SEEDS_GPQA[@]}")
emit_eval_jobs \
  "gpqa_diamond" \
  "$TAG_ENTROPYSHIFT" \
  "$GPQA_TEMP" \
  "$GPQA_TOP_P" \
  "maj@k" \
  "$NUM_SAMPLES" \
  --dynamic_sampling_policy entropy_shift \
  --dynamic_sampling_kwargs "$dyn_kwargs"

echo "Wrote queue jobs to $JOB_FILE"
