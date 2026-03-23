#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/heuristic_finale_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model path differs.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODE="${MODE:-maj@k}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

# Finalist heuristics selected from the general_dev sweep.
HEURISTIC_TEMP="${HEURISTIC_TEMP:-0.798}"
HEURISTIC_TOP_P="${HEURISTIC_TOP_P:-0.907}"
EDT_THETA="${EDT_THETA:-0.2}"
EDT_N="${EDT_N:-0.8}"
ENTROPY_SHIFT_DELTA="${ENTROPY_SHIFT_DELTA:-0.20}"
ENTROPY_MEAN="${ENTROPY_MEAN:-0.07197317484105381}"

TAG_ENTROPYSHIFT="${TAG_ENTROPYSHIFT:-entropyshift-d0.20-r1-distill-qwen7b}"
TAG_EDT="${TAG_EDT:-edt-th0.2-r1-distill-qwen7b}"

DATASETS=(aime24 hmmt25 gpqa_diamond mmlu_pro_lite)
SEEDS_8=(42 43 44 45 46 47 48 49)
SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# heuristic finale queue jobs
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
  local mode="$5"
  local num_samples="$6"
  shift 6
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN"
      utils/llm_eval.py
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

for dataset in "${DATASETS[@]}"; do
  set_dataset_seeds "$dataset"

  entropyshift_kwargs=$(printf '{"T_base": %s, "delta": %s, "H_mean": %s}' \
    "$HEURISTIC_TEMP" "$ENTROPY_SHIFT_DELTA" "$ENTROPY_MEAN")
  emit_eval_jobs \
    "$dataset" \
    "$TAG_ENTROPYSHIFT" \
    "$HEURISTIC_TEMP" \
    "$HEURISTIC_TOP_P" \
    "$MODE" \
    "$NUM_SAMPLES" \
    --dynamic_sampling_policy entropy_shift \
    --dynamic_sampling_kwargs "$entropyshift_kwargs"

  edt_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' \
    "$HEURISTIC_TEMP" "$EDT_THETA" "$EDT_N")
  emit_eval_jobs \
    "$dataset" \
    "$TAG_EDT" \
    "$HEURISTIC_TEMP" \
    "$HEURISTIC_TOP_P" \
    "$MODE" \
    "$NUM_SAMPLES" \
    --dynamic_sampling_policy edt \
    --dynamic_sampling_kwargs "$edt_kwargs"
done

echo "Wrote queue jobs to $JOB_FILE"
