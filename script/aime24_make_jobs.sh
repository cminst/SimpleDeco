#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="aime24"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/aime24_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model paths differ.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-R1-Distill-Qwen-7B-merged}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

DEFAULT_TEMP="${DEFAULT_TEMP:-0.6}"
DEFAULT_TOP_P="${DEFAULT_TOP_P:-0.95}"
MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-0.798}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.907}"

# Set this to the held-out mean normalized entropy once measured.
ENTROPY_MEAN="${ENTROPY_MEAN:-0.5}"
DELTAS=(${DELTAS:-0.10 0.20 0.30})

TAG_BASE="${TAG_BASE:-base-r1-distill-qwen7b}"
TAG_AUTODECO="${TAG_AUTODECO:-autodeco-r1-distill-qwen7b}"
TAG_GREEDY="${TAG_GREEDY:-greedy-r1-distill-qwen7b}"
TAG_MEANSHIFT="${TAG_MEANSHIFT:-meanshift2-r1-distill-qwen7b}"
TAG_ENTROPY_SHIFT_PREFIX="${TAG_ENTROPY_SHIFT_PREFIX:-entropyshift-r1-distill-qwen7b}"

SEEDS_8=(42 43 44 45 46 47 48 49)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# aime24 queue jobs
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
    local -a cmd=(
      "$PYTHON_BIN" utils/llm_eval.py \
      --model_name_or_path "$model" \
      --dataset "$DATASET" \
      --temp "$temp" \
      --top_p "$top_p" \
      --mode "$mode" \
      --num_samples "$num_samples" \
      --tp_size "$TP_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --seed "$seed" \
      --save_outputs "$out"
    )
    if ((${#extra_args[@]} > 0)); then
      cmd+=("${extra_args[@]}")
    fi
    emit_job "$out" "$log" "${cmd[@]}"
  done
}

SEEDS=("${SEEDS_8[@]}")

# 1) Default operating point.
emit_eval_jobs "$TAG_BASE" "$MODEL_BASE" "$DEFAULT_TEMP" "$DEFAULT_TOP_P" "maj@k" "$NUM_SAMPLES"

# 2) MeanShift anchor for the adaptive sweep.
emit_eval_jobs "$TAG_MEANSHIFT" "$MODEL_BASE" "$MEANSHIFT_TEMP" "$MEANSHIFT_TOP_P" "maj@k" "$NUM_SAMPLES"

# 3) EntropyShift sweep around the MeanShift temperature.
for delta in "${DELTAS[@]}"; do
  tag="${TAG_ENTROPY_SHIFT_PREFIX}-d${delta}-hmean${ENTROPY_MEAN}"
  dyn_kwargs=$(printf '{"T_base": %s, "delta": %s, "H_mean": %s}' \
    "$MEANSHIFT_TEMP" "$delta" "$ENTROPY_MEAN")
  emit_eval_jobs \
    "$tag" \
    "$MODEL_BASE" \
    "$MEANSHIFT_TEMP" \
    "$MEANSHIFT_TOP_P" \
    "maj@k" \
    "$NUM_SAMPLES" \
    --dynamic_sampling_policy entropy_shift \
    --dynamic_sampling_kwargs "$dyn_kwargs"
done

# 4) AutoDeco learned controller.
emit_eval_jobs "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 1.0 "maj@k" "$NUM_SAMPLES"

# 5) Greedy reference: one seed, one sample.
SEEDS=(42)
emit_eval_jobs "$TAG_GREEDY" "$MODEL_BASE" 0.0 0.95 "maj@k" 1

echo "Wrote queue jobs to $JOB_FILE"
