#!/usr/bin/env bash
set -euo pipefail

# Generic EDT-only queue job generator.
#
# Typical usage for the current R1-Distill-Qwen-7B setup:
#   1) Theta selection on the mixed dev slice:
#      bash script/edt_make_jobs.sh
#   2) MMLU-Pro-lite sanity check after freezing theta from the dev run:
#      DATASET=mmlu_pro_lite \
#      JOB_FILE=jobs/edt_mmlu_pro_lite_jobs.txt \
#      SEEDS="42 43 44 45 46 47 48 49" \
#      NUM_SAMPLES=16 \
#      EDT_THETAS="0.1" \
#      bash script/edt_make_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="${DATASET:-general_dev}"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/edt_${DATASET}_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-maj@k}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

SEEDS_RAW="${SEEDS:-42 43}"
ANCHOR_NAME="${ANCHOR_NAME:-meanshift}"
ANCHOR_TEMP="${ANCHOR_TEMP:-0.798}"
ANCHOR_TOP_P="${ANCHOR_TOP_P:-0.907}"
EDT_N="${EDT_N:-0.8}"
EDT_THETAS_RAW="${EDT_THETAS:-0.1 0.3}"
TAG_PREFIX="${TAG_PREFIX:-edt-r1-distill-qwen7b}"

INCLUDE_PAPER_SANITY="${INCLUDE_PAPER_SANITY:-0}"
PAPER_T0="${PAPER_T0:-0.6}"
PAPER_THETA="${PAPER_THETA:-0.1}"
PAPER_N="${PAPER_N:-0.8}"
PAPER_TOP_P="${PAPER_TOP_P:-0.95}"

read -r -a SEED_ARRAY <<< "$SEEDS_RAW"
read -r -a EDT_THETA_ARRAY <<< "$EDT_THETAS_RAW"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "SEEDS must contain at least one seed." >&2
  exit 1
fi

if [[ "${#EDT_THETA_ARRAY[@]}" -eq 0 ]]; then
  echo "EDT_THETAS must contain at least one theta." >&2
  exit 1
fi

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# ${DATASET} EDT queue jobs
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
  local temp="$2"
  local top_p="$3"
  shift 3
  local -a extra_args=("$@")

  for seed in "${SEED_ARRAY[@]}"; do
    local out="ckpt/${DATASET}/${tag}/maj${NUM_SAMPLES}_seed${seed}.jsonl"
    local log="ckpt/${DATASET}/${tag}/maj${NUM_SAMPLES}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN" utils/llm_eval.py
      --model_name_or_path "$MODEL_BASE"
      --dataset "$DATASET"
      --temp "$temp"
      --top_p "$top_p"
      --mode "$MODE"
      --num_samples "$NUM_SAMPLES"
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

for theta in "${EDT_THETA_ARRAY[@]}"; do
  tag="${TAG_PREFIX}-${ANCHOR_NAME}-t${ANCHOR_TEMP}-p${ANCHOR_TOP_P}-th${theta}-n${EDT_N}"
  dyn_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' \
    "$ANCHOR_TEMP" "$theta" "$EDT_N")
  emit_eval_jobs \
    "$tag" \
    "$ANCHOR_TEMP" \
    "$ANCHOR_TOP_P" \
    --dynamic_sampling_policy edt \
    --dynamic_sampling_kwargs "$dyn_kwargs"
done

if [[ "$INCLUDE_PAPER_SANITY" == "1" ]]; then
  paper_tag="${TAG_PREFIX}-paperdefault-t${PAPER_T0}-p${PAPER_TOP_P}-th${PAPER_THETA}-n${PAPER_N}"
  paper_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' \
    "$PAPER_T0" "$PAPER_THETA" "$PAPER_N")
  emit_eval_jobs \
    "$paper_tag" \
    "$PAPER_T0" \
    "$PAPER_TOP_P" \
    --dynamic_sampling_policy edt \
    --dynamic_sampling_kwargs "$paper_kwargs"
fi

echo "Wrote queue jobs to $JOB_FILE"
