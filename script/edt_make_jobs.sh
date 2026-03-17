#!/usr/bin/env bash
set -euo pipefail

# Generic EDT queue job generator.
#
# Typical usage for the current R1-Distill-Qwen-7B setup:
#   1) General-dev sweep:
#      - EDT on 4 seeds across theta candidates
#      - 2 extra seeds for each static-dev baseline
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
if [[ "$DATASET" == "general_dev" ]]; then
  MODE_DEFAULT="pass@k"
  SEEDS_DEFAULT="42 43 44 45"
else
  MODE_DEFAULT="maj@k"
  SEEDS_DEFAULT="42 43"
fi
MODE="${MODE:-$MODE_DEFAULT}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

SEEDS_RAW="${SEEDS:-$SEEDS_DEFAULT}"
ANCHOR_NAME="${ANCHOR_NAME:-meanshift}"
ANCHOR_TEMP="${ANCHOR_TEMP:-0.798}"
ANCHOR_TOP_P="${ANCHOR_TOP_P:-0.907}"
EDT_N="${EDT_N:-0.8}"
EDT_THETAS_RAW="${EDT_THETAS:-0.1 0.3}"
TAG_PREFIX="${TAG_PREFIX:-edt-r1-distill-qwen7b}"

INCLUDE_STATIC_BASELINES="${INCLUDE_STATIC_BASELINES:-auto}"
STATIC_SEEDS_RAW="${STATIC_SEEDS:-44 45}"
STATIC_TEMPS_RAW="${STATIC_TEMPS:-0.7 0.8 0.9}"
STATIC_TOP_PS_RAW="${STATIC_TOP_PS:-0.90 0.95}"
STATIC_TAG_PREFIX="${STATIC_TAG_PREFIX:-static-dev}"
STATIC_MODE="${STATIC_MODE:-$MODE}"
STATIC_NUM_SAMPLES="${STATIC_NUM_SAMPLES:-$NUM_SAMPLES}"

INCLUDE_PAPER_SANITY="${INCLUDE_PAPER_SANITY:-0}"
PAPER_T0="${PAPER_T0:-0.6}"
PAPER_THETA="${PAPER_THETA:-0.1}"
PAPER_N="${PAPER_N:-0.8}"
PAPER_TOP_P="${PAPER_TOP_P:-0.95}"

read -r -a EDT_THETA_ARRAY <<< "$EDT_THETAS_RAW"

if [[ "$INCLUDE_STATIC_BASELINES" == "auto" ]]; then
  if [[ "$DATASET" == "general_dev" ]]; then
    INCLUDE_STATIC_BASELINES="1"
  else
    INCLUDE_STATIC_BASELINES="0"
  fi
fi

if [[ -z "$SEEDS_RAW" ]]; then
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
  local model="$2"
  local temp="$3"
  local top_p="$4"
  local mode="$5"
  local num_samples="$6"
  local seeds_raw="$7"
  shift 7
  local -a extra_args=("$@")
  local -a seeds
  read -r -a seeds <<< "$seeds_raw"

  if [[ "${#seeds[@]}" -eq 0 ]]; then
    echo "emit_eval_jobs received an empty seed list for tag ${tag}." >&2
    exit 1
  fi

  for seed in "${seeds[@]}"; do
    local out="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${DATASET}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN" utils/llm_eval.py
      --model_name_or_path "$model"
      --dataset "$DATASET"
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

for theta in "${EDT_THETA_ARRAY[@]}"; do
  tag="${TAG_PREFIX}-${ANCHOR_NAME}-t${ANCHOR_TEMP}-p${ANCHOR_TOP_P}-th${theta}-n${EDT_N}"
  dyn_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' \
    "$ANCHOR_TEMP" "$theta" "$EDT_N")
  emit_eval_jobs \
    "$tag" \
    "$MODEL_BASE" \
    "$ANCHOR_TEMP" \
    "$ANCHOR_TOP_P" \
    "$MODE" \
    "$NUM_SAMPLES" \
    "$SEEDS_RAW" \
    --dynamic_sampling_policy edt \
    --dynamic_sampling_kwargs "$dyn_kwargs"
done

if [[ "$INCLUDE_STATIC_BASELINES" == "1" ]]; then
  read -r -a static_temps <<< "$STATIC_TEMPS_RAW"
  read -r -a static_top_ps <<< "$STATIC_TOP_PS_RAW"

  for temp in "${static_temps[@]}"; do
    for top_p in "${static_top_ps[@]}"; do
      static_tag="${STATIC_TAG_PREFIX}-t${temp}-p${top_p}"
      emit_eval_jobs \
        "$static_tag" \
        "$MODEL_BASE" \
        "$temp" \
        "$top_p" \
        "$STATIC_MODE" \
        "$STATIC_NUM_SAMPLES" \
        "$STATIC_SEEDS_RAW"
    done
  done
fi

if [[ "$INCLUDE_PAPER_SANITY" == "1" ]]; then
  paper_tag="${TAG_PREFIX}-paperdefault-t${PAPER_T0}-p${PAPER_TOP_P}-th${PAPER_THETA}-n${PAPER_N}"
  paper_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' \
    "$PAPER_T0" "$PAPER_THETA" "$PAPER_N")
  emit_eval_jobs \
    "$paper_tag" \
    "$MODEL_BASE" \
    "$PAPER_T0" \
    "$PAPER_TOP_P" \
    "$MODE" \
    "$NUM_SAMPLES" \
    "$SEEDS_RAW" \
    --dynamic_sampling_policy edt \
    --dynamic_sampling_kwargs "$paper_kwargs"
fi

echo "Wrote queue jobs to $JOB_FILE"
