# Base model
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/qwen3-4b-thinking \
#   --dataset aime24 \
#   --temp 0.6 \
#   --top_p 0.95 \
#   --k 16 \
#   --seed 42 \
#   --tp_size 1

python utils/llm_eval.py \
  --model_name_or_path ckpt/autodeco-qwen3-4b-thinking \
  --dataset aime24 \
  --temp 1.0 \
  --top_p 0.95 \
  --k 16 \
  --seed 42 \
  --tp_size 1