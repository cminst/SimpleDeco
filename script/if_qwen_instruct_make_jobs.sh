#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/if_qwen3_30b_instruct_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

RESULT_ROOT="${RESULT_ROOT:-ckpt_qwen_instruct}"
# Base/meanshift/greedy all use the same merged checkpoint with heads disabled.
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-Qwen3-30B-Instruct-Merged}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

BASE_TEMP="${BASE_TEMP:-0.7}"
BASE_TOP_P="${BASE_TOP_P:-0.8}"
MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-0.861}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.843}"
TAG_BASE="${TAG_BASE:-base-qwen3-30b-instruct}"
TAG_AUTODECO="${TAG_AUTODECO:-autodeco-qwen3-30b-instruct}"
TAG_GREEDY="${TAG_GREEDY:-greedy-qwen3-30b-instruct}"
TAG_MEANSHIFT="${TAG_MEANSHIFT:-meanshift-qwen3-30b-instruct}"

# 16 seeds for stochastic methods (mean ± 95% CI reporting).
# Greedy is deterministic so it only gets one seed.
SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)

DATASETS=("ifeval" "ifbench")

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# if_eval queue jobs — Qwen3-30B-Instruct (IFEval + IFBench)
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

# emit_if_jobs TAG TEMP TOP_P SEEDS_ARRAY_NAME [EXTRA_ARGS...]
#   Pass "SINGLE_SEED" as SEEDS_ARRAY_NAME to run only seed 42 (e.g. greedy).
emit_if_jobs() {
  local tag="$1"
  local temp="$2"
  local top_p="$3"
  local seeds_var="$4"
  shift 4
  local -a extra_args=("$@")

  local -a seeds
  if [[ "$seeds_var" == "SINGLE_SEED" ]]; then
    seeds=(42)
  else
    eval "seeds=(\"\${${seeds_var}[@]}\")"
  fi

  for dataset in "${DATASETS[@]}"; do
    for seed in "${seeds[@]}"; do
      local out="${RESULT_ROOT}/${dataset}/${tag}/if_eval_seed${seed}.jsonl"
      local log="${RESULT_ROOT}/${dataset}/${tag}/if_eval_seed${seed}.log"
      local -a cmd=(
        "$PYTHON_BIN" utils/if_eval.py
        --model_name_or_path "$MODEL_AUTODECO"
        --dataset "$dataset"
        --temp "$temp"
        --top_p "$top_p"
        --tp_size "$TP_SIZE"
        --max_tokens "$MAX_TOKENS"
        --seed "$seed"
        --strip_think
        --output-file "$out"
      )
      if ((${#extra_args[@]} > 0)); then
        cmd+=("${extra_args[@]}")
      fi
      emit_job "$out" "$log" "${cmd[@]}"
    done
  done
}

# 1) Base operating point — 16 seeds for CI.
emit_if_jobs "$TAG_BASE" "$BASE_TEMP" "$BASE_TOP_P" SEEDS --autodeco_heads none

# 2) Greedy reference — deterministic, single seed.
emit_if_jobs "$TAG_GREEDY" 0.0 "$BASE_TOP_P" SINGLE_SEED --autodeco_heads none

# 3) AutoDeco learned controller — 16 seeds for CI.
emit_if_jobs "$TAG_AUTODECO" 1.0 1.0 SEEDS --autodeco_heads temperature,top_p

# 4) MeanShift (fixed operating point at the train-split mean temperature) — 16 seeds for CI.
emit_if_jobs "$TAG_MEANSHIFT" "$MEANSHIFT_TEMP" "$MEANSHIFT_TOP_P" SEEDS --autodeco_heads none

echo "Wrote $(grep -c 'fi$' "$JOB_FILE" || true) jobs to $JOB_FILE"
