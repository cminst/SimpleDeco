# --- Base model ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/qwen3-4b-thinking \
#   --dataset aime24 \
#   --temp 0.6 \
#   --top_p 0.95 \
#   --k 16 \
#   --seed 42 \
#   --tp_size 1

# --- AutoDeco ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/autodeco-qwen3-4b-thinking \
#   --dataset aime24 \
#   --temp 1.0 \
#   --top_p 0.95 \
#   --k 16 \
#   --seed 42 \
#   --tp_size 1

# --- SimpleDeco ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/simpledeco-qwen3-4b-thinking-merged \
#   --dataset aime24 \
#   --temp 1.0 \
#   --top_p 0.95 \
#   --k 16 \
#   --seed 43 \
#   --tp_size 4

# --- No Top-P Guardrails ---
python utils/llm_eval.py \
  --model_name_or_path ckpt/simpledeco-qwen3-4b-thinking-merged \
  --dataset aime24 \
  --temp 1.0 \
  --top_p 1.0 \
  --top_k -1 \
  --k 16 \
  --mode maj@k \
  --seed 42 \
  --tp_size 1  2>&1 | tee ckpt/simpledeco_aime24_handsoff.log