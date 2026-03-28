#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/if_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

# Update if your model paths differ.
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_AUTODECO="${MODEL_AUTODECO:-ckpt/AutoDeco-R1-Distill-Qwen-7B-merged}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

DEFAULT_TEMP="${DEFAULT_TEMP:-0.6}"
DEFAULT_TOP_P="${DEFAULT_TOP_P:-0.95}"
MEANSHIFT_TEMP="${MEANSHIFT_TEMP:-0.798}"
MEANSHIFT_TOP_P="${MEANSHIFT_TOP_P:-0.907}"
TAG_BASE="${TAG_BASE:-base-r1-distill-qwen7b}"
TAG_AUTODECO="${TAG_AUTODECO:-autodeco-r1-distill-qwen7b}"
TAG_GREEDY="${TAG_GREEDY:-greedy-r1-distill-qwen7b}"
TAG_MEANSHIFT="${TAG_MEANSHIFT:-meanshift-r1-distill-qwen7b}"

# 8 seeds for stochastic methods (mean ± 95% CI reporting).
# Greedy is deterministic so it only gets one seed.
SEEDS=(42 43 44 45 46 47 48 49)

DATASETS=("ifeval" "ifbench")

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF
# if_eval queue jobs (IFEval + IFBench)
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

# emit_if_jobs TAG MODEL TEMP TOP_P [SEEDS_ARRAY_NAME] [EXTRA_ARGS...]
#   SEEDS_ARRAY_NAME  name of a bash array variable listing seeds to run.
#                     Pass "SINGLE_SEED" to run only seed 42 (e.g. greedy).
emit_if_jobs() {
  local tag="$1"
  local model="$2"
  local temp="$3"
  local top_p="$4"
  local seeds_var="$5"
  shift 5
  local -a extra_args=("$@")

  # Resolve the seeds array by name.
  local -a seeds
  if [[ "$seeds_var" == "SINGLE_SEED" ]]; then
    seeds=(42)
  else
    eval "seeds=(\"\${${seeds_var}[@]}\")"
  fi

  for dataset in "${DATASETS[@]}"; do
    for seed in "${seeds[@]}"; do
      local out="ckpt/${dataset}/${tag}/if_eval_seed${seed}.jsonl"
      local log="ckpt/${dataset}/${tag}/if_eval_seed${seed}.log"
      local -a cmd=(
        "$PYTHON_BIN" utils/if_eval.py
        --model_name_or_path "$model"
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

# 1) Base operating point — 8 seeds for CI.
emit_if_jobs "$TAG_BASE" "$MODEL_BASE" "$DEFAULT_TEMP" "$DEFAULT_TOP_P" SEEDS

# 2) Greedy reference — deterministic, single seed.
emit_if_jobs "$TAG_GREEDY" "$MODEL_BASE" 0.0 "$DEFAULT_TOP_P" SINGLE_SEED

# 3) AutoDeco learned controller — 8 seeds for CI.
emit_if_jobs "$TAG_AUTODECO" "$MODEL_AUTODECO" 1.0 1.0 SEEDS --autodeco_heads temperature top_p

# 4) MeanShift (fixed operating point at the train-split mean temperature) — 8 seeds for CI.
emit_if_jobs "$TAG_MEANSHIFT" "$MODEL_BASE" "$MEANSHIFT_TEMP" "$MEANSHIFT_TOP_P" SEEDS

echo "Wrote $(grep -c 'fi$' "$JOB_FILE" || true) jobs to $JOB_FILE"
