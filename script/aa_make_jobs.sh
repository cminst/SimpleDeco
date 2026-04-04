#!/usr/bin/env bash
# Generate evaluation jobs for Analytic Alignment across 4 benchmarks.
# Adds AA and AA-MeanShift methods to complement existing Base/AutoDeco/MeanShift jobs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FILTER_EXISTING="${FILTER_EXISTING:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_AA="${MODEL_AA:-ckpt/AnalyticAlign-R1-Distill-Qwen-7B-merged}"
MODEL_BASE="${MODEL_BASE:-ckpt/DeepSeek-R1-Distill-Qwen-7B}"

# AA-MeanShift: mean T_hat from AA diagnostics on val_balanced
AA_MEANSHIFT_TEMP="${AA_MEANSHIFT_TEMP:-0.746}"
AA_MEANSHIFT_TOP_P="${AA_MEANSHIFT_TOP_P:-0.95}"

TAG_AA="${TAG_AA:-aa-r1-distill-qwen7b}"
TAG_AA_MEANSHIFT="${TAG_AA_MEANSHIFT:-aa-meanshift-r1-distill-qwen7b}"

TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"

SEEDS_8=(42 43 44 45 46 47 48 49)
SEEDS_16=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57)

emit_job() {
  local job_file="$1"
  local out="$2"
  local log="$3"
  local out_dir
  out_dir="$(dirname "$out")"
  if [[ "$FILTER_EXISTING" == "1" && -s "$out" ]]; then
    return
  fi
  shift 3
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
    "$out_q" "$out" "$out_dir_q" "$cmd_str" "$log_q" >> "$job_file"
}

emit_eval_jobs() {
  local job_file="$1"
  local dataset="$2"
  local tag="$3"
  local model="$4"
  local temp="$5"
  local top_p="$6"
  local mode="$7"
  local num_samples="$8"
  shift 8
  local -a extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.jsonl"
    local log="ckpt/${dataset}/${tag}/maj${num_samples}_seed${seed}.log"
    local -a cmd=(
      "$PYTHON_BIN" utils/llm_eval.py \
      --model_name_or_path "$model" \
      --dataset "$dataset" \
      --temp "$temp" \
      --top_p "$top_p" \
      --mode "$mode" \
      --num_samples "$num_samples" \
      --tp_size "$TP_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --seed "$seed" \
      --output-file "$out"
    )
    if ((${#extra_args[@]} > 0)); then
      cmd+=("${extra_args[@]}")
    fi
    emit_job "$job_file" "$out" "$log" "${cmd[@]}"
  done
}

DATASETS=(aime24 gpqa_diamond hmmt25 mmlu_pro_lite)

for dataset in "${DATASETS[@]}"; do
  job_file="$ROOT_DIR/jobs/aa_${dataset}_jobs.txt"
  mkdir -p "$(dirname "$job_file")"

  # HMMT25 uses 16 seeds; all others use 8
  if [[ "$dataset" == "hmmt25" ]]; then
    SEEDS=("${SEEDS_16[@]}")
  else
    SEEDS=("${SEEDS_8[@]}")
  fi

  cat > "$job_file" <<EOF
# ${dataset} AA queue jobs
EOF

  # 1) AA: learned temperature head, top_p passthrough at 0.95
  #    Model has temp head; pass temp=1.0 so vLLM uses head predictions.
  #    top_p=0.95 since AA has no top_p head.
  emit_eval_jobs "$job_file" "$dataset" "$TAG_AA" "$MODEL_AA" 1.0 0.95 "maj@k" "$NUM_SAMPLES" \
    --autodeco_heads temperature

  # 2) AA-MeanShift: fixed at AA's mean T, no heads
  emit_eval_jobs "$job_file" "$dataset" "$TAG_AA_MEANSHIFT" "$MODEL_BASE" "$AA_MEANSHIFT_TEMP" "$AA_MEANSHIFT_TOP_P" "maj@k" "$NUM_SAMPLES"

  echo "Wrote ${dataset} AA jobs to $job_file"
done
