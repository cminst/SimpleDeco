#!/bin/bash
# bash script/test_generation_tp4.sh aime24 1.0 0.99 -1 1.0 16 ckpt/simpledeco-qwen3-4b-thinking-merged/ 2>&1 | tee run_aime24_tp4.log
DATASET=${1:-aime24}
TEMP=${2:-1.0}
TOP_P=${3:-1.0}
TOP_K=${4:--1}
RP=${5:-1.0}
K=${6:-16}
MODEL_NAME_OR_PATH=${7:-path_to_your_model}

echo "Using parameters:"
echo "  DATASET: $DATASET"
echo "  TEMP: $TEMP"
echo "  TOP_P: $TOP_P"
echo "  TOP_K: $TOP_K"
echo "  RP: $RP"
echo "  K: $K"
echo "  MODEL: $MODEL_NAME_OR_PATH"
echo "  TP_SIZE: 1"
echo ""

echo "Generating 8 random seeds and running llm_eval with TP=1..."
seeds=()
for i in {1..8}; do
    seed=$((RANDOM % 999999 + 1))
    seeds+=($seed)
    echo "Generated seed $i: $seed"
done

echo ""
echo "All seeds: ${seeds[*]}"
echo ""

echo "Starting tasks in batches of 4 (each uses 1 GPU)..."

pids=()
for i in {0..7}; do
    batch_slot=$((i % 4))
    seed=${seeds[$i]}
    
    case $batch_slot in
        0) gpus="0" ;;
        1) gpus="1" ;;
        2) gpus="2" ;;
        3) gpus="3" ;;
    esac
    
    echo "Starting task $((i+1)) with seed $seed on GPU $gpus (TP=1)"
    CUDA_VISIBLE_DEVICES=$gpus python utils/llm_eval.py \
        --k $K \
        --temp $TEMP \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --rp $RP \
        --seed $seed \
        --dataset $DATASET \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tp_size 1 &
    pids+=($!)

    if [ $batch_slot -eq 3 ]; then
        wait ${pids[@]}
        pids=()
        echo "Batch completed."
    fi
done

# Wait for any remaining tasks if the loop ends early (not needed for exactly 8 tasks, but safe)
wait
echo "All tasks completed."