#!/usr/bin/env bash
# Generates all 8 seeds (42-49) for all static-dev configs on general_dev.
# Already-completed seeds are skipped by FILTER_EXISTING, so running this
# on eagle3 will only execute whichever seeds are missing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="general_dev"
JOB_FILE="${JOB_FILE:-$ROOT_DIR/jobs/general_dev_static_extra_seeds_jobs.txt}"
APPEND="${APPEND:-0}"
FILTER_EXISTING="${FILTER_EXISTING:-1}"

MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"

NUM_SAMPLES="${NUM_SAMPLES:-8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

MODES=(${MODES:-pass@k})

SEEDS=(42 43 44 45 46 47 48 49)

STATIC_TEMPS=(0.7 0.8 0.9)
STATIC_TOP_PS=(0.90 0.95)
TAG_PREFIX_STATIC="${TAG_PREFIX_STATIC:-static-dev}"

mkdir -p "$(dirname "$JOB_FILE")"

if [[ "$APPEND" != "1" ]]; then
  cat > "$JOB_FILE" <<EOF2
# general_dev static seeds (42-49) queue jobs
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
  local out_q log_q out_dir_q
  out_q=$(printf "%q" "$out")
  log_q=$(printf "%q" "$log")
  out_dir_q=$(printf "%q" "$out_dir")
  printf "if [ -s %s ]; then echo \"Skipping existing %s\"; else mkdir -p %s; %s2>&1 | tee %s; fi\n" \
    "$out_q" "$out" "$out_dir_q" "$cmd_str" "$log_q" >> "$JOB_FILE"
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
      for seed in "${SEEDS[@]}"; do
        out="ckpt/${DATASET}/${tag}/maj${NUM_SAMPLES}_seed${seed}.jsonl"
        log="ckpt/${DATASET}/${tag}/maj${NUM_SAMPLES}_seed${seed}.log"
        emit_job "$out" "$log" \
          python utils/llm_eval.py \
          --model_name_or_path "$MODEL_BASE" \
          --dataset "$DATASET" \
          --temp "$temp" \
          --top_p "$top_p" \
          --mode "$mode" \
          --num_samples "$NUM_SAMPLES" \
          --tp_size "$TP_SIZE" \
          --max_tokens "$MAX_TOKENS" \
          --seed "$seed" \
          --output-file "$out"
      done
    done
  done
done

echo "Wrote queue jobs to $JOB_FILE"
