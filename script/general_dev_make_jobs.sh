#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="general_dev"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/general_dev_sweep_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model path differs.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"

# Sweep settings (override via env vars as needed).
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

# Space-delimited list of modes, e.g. "pass@k" or "pass@k maj@k".
MODES=(${MODES:-pass@k})

SEEDS_2=(42 43)
SEEDS=("${SEEDS_2[@]}")
EXTRA_SEEDS_6=(${EXTRA_SEEDS_6:-44 45 46 47 48 49})

# Static baseline sweep grid.
STATIC_TEMPS=(0.7 0.8 0.9)
STATIC_TOP_PS=(0.90 0.95)
TAG_PREFIX_STATIC="${TAG_PREFIX_STATIC:-static-dev}"

# EDT sweep anchored at the repo's MeanShift operating point.
EDT_T0="${EDT_T0:-0.798}"
EDT_TOP_P="${EDT_TOP_P:-0.907}"
EDT_N="${EDT_N:-0.8}"
EDT_THETAS=(${EDT_THETAS:-0.1 0.2 0.3})
TAG_PREFIX_EDT="${TAG_PREFIX_EDT:-edt-r1-distill-qwen7b-meanshift-t${EDT_T0}-p${EDT_TOP_P}}"

# ConfGate sweep (set thresholds to your target sample-rate quantiles).
CONF_T_HIGH="${CONF_T_HIGH:-0.9}"
CONF_THRESHOLDS=(${CONF_THRESHOLDS:-0.248 0.354 0.441 0.518 0.591})
TAG_PREFIX_CONFGATE="${TAG_PREFIX_CONFGATE:-confgate-dev}"

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF2
# general_dev sweep queue jobs
EOF2
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
    emit_job "$out" "$log" \
      python utils/llm_eval.py \
      --model_name_or_path "$model" \
      --dataset "$DATASET" \
      --temp "$temp" \
      --top_p "$top_p" \
      --mode "$mode" \
      --num_samples "$num_samples" \
      --tp_size "$TP_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --seed "$seed" \
      --save_outputs "$out" \
      "${extra_args[@]}"
  done
}

mode_tag_for() {
  local mode="$1"
  mode="${mode//@/}"
  mode="${mode//[^a-zA-Z0-9]/}"
  printf "%s" "$mode"
}

for temp in "${STATIC_TEMPS[@]}"; do
  for top_p in "${STATIC_TOP_PS[@]}"; do
    for mode in "${MODES[@]}"; do
      tag="${TAG_PREFIX_STATIC}-t${temp}-p${top_p}"
      if [[ "${#MODES[@]}" -gt 1 ]]; then
        tag="${tag}-$(mode_tag_for "$mode")"
      fi
      emit_eval_jobs "$tag" "$MODEL_BASE" "$temp" "$top_p" "$mode" "$NUM_SAMPLES"
    done
  done
done

for theta in "${EDT_THETAS[@]}"; do
  for mode in "${MODES[@]}"; do
    tag="${TAG_PREFIX_EDT}-th${theta}-n${EDT_N}"
    if [[ "${#MODES[@]}" -gt 1 ]]; then
      tag="${tag}-$(mode_tag_for "$mode")"
    fi
    dyn_kwargs=$(printf '{"T0": %s, "theta": %s, "N": %s}' "$EDT_T0" "$theta" "$EDT_N")
    emit_eval_jobs \
      "$tag" \
      "$MODEL_BASE" \
      "$EDT_T0" \
      "$EDT_TOP_P" \
      "$mode" \
      "$NUM_SAMPLES" \
      --dynamic_sampling_policy edt \
      --dynamic_sampling_kwargs "$dyn_kwargs"
  done
done

for threshold in "${CONF_THRESHOLDS[@]}"; do
  for mode in "${MODES[@]}"; do
    tag="${TAG_PREFIX_CONFGATE}-tau${threshold}-Thigh${CONF_T_HIGH}"
    if [[ "${#MODES[@]}" -gt 1 ]]; then
      tag="${tag}-$(mode_tag_for "$mode")"
    fi
    dyn_kwargs=$(printf '{"maxprob_threshold": %s, "T_high": %s}' "$threshold" "$CONF_T_HIGH")
    emit_eval_jobs \
      "$tag" \
      "$MODEL_BASE" \
      1.0 \
      0.95 \
      "$mode" \
      "$NUM_SAMPLES" \
      --dynamic_sampling_policy confidence_gated \
      --dynamic_sampling_kwargs "$dyn_kwargs"
  done
done

# Extra seeds for specific configs to reach 8 total.
if [[ "${#EXTRA_SEEDS_6[@]}" -gt 0 ]]; then
  SEEDS=("${EXTRA_SEEDS_6[@]}")

  # Static baseline: temp 0.7, top-p 0.90.
  for mode in "${MODES[@]}"; do
    tag="${TAG_PREFIX_STATIC}-t0.7-p0.9"
    if [[ "${#MODES[@]}" -gt 1 ]]; then
      tag="${tag}-$(mode_tag_for "$mode")"
    fi
    emit_eval_jobs "$tag" "$MODEL_BASE" 0.7 0.90 "$mode" "$NUM_SAMPLES"
  done

  # ConfGate: tau 0.518 and 0.441 with T_high=0.9.
  for threshold in 0.518 0.441; do
    for mode in "${MODES[@]}"; do
      tag="${TAG_PREFIX_CONFGATE}-tau${threshold}-Thigh${CONF_T_HIGH}"
      if [[ "${#MODES[@]}" -gt 1 ]]; then
        tag="${tag}-$(mode_tag_for "$mode")"
      fi
      dyn_kwargs=$(printf '{"maxprob_threshold": %s, "T_high": %s}' "$threshold" "$CONF_T_HIGH")
      emit_eval_jobs \
        "$tag" \
        "$MODEL_BASE" \
        1.0 \
        0.95 \
        "$mode" \
        "$NUM_SAMPLES" \
        --dynamic_sampling_policy confidence_gated \
        --dynamic_sampling_kwargs "$dyn_kwargs"
    done
  done
fi

echo "Wrote queue jobs to $JOB_FILE"
