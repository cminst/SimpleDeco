# --- Base model ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/Qwen3-4B-Thinking-2507 \
#   --dataset aime24 \
#   --temp 0.6 \
#   --top_p 0.95 \
#   --rp 1.05 \
#   --mode maj@k \
#   --num_samples 8 \
#   --seed 42 \
#   --tp_size 1 \
#   --max_tokens 32768 \
#   --save_outputs ckpt/qwen3_4b_thinking_aime24_maj8_seed42.jsonl 2>&1 | tee ckpt/aime24/qwen3-4b-thinking/maj8_seed42.log

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
#   --model_name_or_path ckpt/simpledeco-ql-nosmooth-merged \
#   --dataset aime24 \
#   --temp 0.1 \
#   --top_p 0.95 \
#   --rp 1.05 \
#   --mode maj@k \
#   --num_samples 8 \
#   --seed 42 \
#   --tp_size 1 \
#   --max_tokens 32768 \
#   --save_outputs ckpt/simpledeco_ql_aime24_maj8_seed42.jsonl 2>&1 | tee ckpt/simpledeco_ql_aime24_maj8_seed42.log

# --- No Top-P Guardrails ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/simpledeco-qwen3-4b-thinking-merged \
#   --dataset aime24 \
#   --temp 1.0 \
#   --top_p 1.0 \
#   --top_k -1 \
#   --k 16 \
#   --mode maj@k \
#   --seed 42 \
#   --tp_size 1  2>&1 | tee ckpt/simpledeco_aime24_handsoff.log

# --- AutoDeco-R1-Distill-Qwen-7B ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/AutoDeco-R1-Distill-Qwen-7B-merged \
#   --dataset aime24 \
#   --temp 1.0 \
#   --top_p 1.0 \
#   --mode maj@k \
#   --num_samples 16 \
#   --seed 42 \
#   --tp_size 1 \
#   --max_tokens 32768 \
#   --save_outputs ckpt/autodeco_r1_distill_qwen7b_aime24_maj16_seed42.jsonl 2>&1 | tee ckpt/aime24/autodeco-r1-distill-qwen7b/maj16_seed42.log

# --- R1 distill default params ---
# python utils/llm_eval.py \
#   --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
#   --dataset aime24 \
#   --temp 0.6 \
#   --top_p 0.95 \
#   --mode maj@k \
#   --num_samples 16 \
#   --seed 42 \
#   --tp_size 1 \
#   --max_tokens 32768 \
#   --save_outputs ckpt/base_r1_distill_qwen7b_aime24_maj16_seed42.jsonl 2>&1 | tee ckpt/aime24/base-r1-distill-qwen7b/maj16_seed42.log

# Meanshift
for seed in 42 43 44 45 46 47 48 49; do  
  python utils/llm_eval.py \
    --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
    --dataset aime24 \
    --temp 0.798 \
    --top_p 0.907 \
    --mode maj@k \
    --num_samples 16 \
    --seed $seed \
    --tp_size 1 \
    --max_tokens 32768 \
    --save_outputs ckpt/meanshift2_r1_distill_qwen7b_aime24_maj16_seed$seed.jsonl 2>&1 | tee ckpt/aime24/meanshift2-r1-distill-qwen7b/maj16_seed$seed.log
done

# for seed in 42 43 44 45 46 47 48 49; do
#   python utils/llm_eval.py \
#     --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset aime24 \
#     --temp 1.0 \
#     --top_p 0.95 \
#     --mode pass@k \
#     --num_samples 16 \
#     --tp_size 1 \
#     --max_tokens 32768 \
#     --dynamic_sampling_policy confidence_gated \
#     --dynamic_sampling_kwargs '{"maxprob_threshold": 0.7, "T_high": 0.9}' \
#     --seed $seed \
#     --save_outputs ckpt/confgate_0.9_0.7_r1_distill_qwen7b_aime24_maj16_seed$seed.jsonl \
#     2>&1 | tee ckpt/aime24/confgate_0.9_0.7-r1-distill-qwen7b/maj16_seed$seed.log
# done

# for seed in 42; do
#   python utils/llm_eval.py \
#     --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset aime24 \
#     --temp 1.0 \
#     --top_p 0.95 \
#     --mode pass@k \
#     --num_samples 16 \
#     --tp_size 1 \
#     --max_tokens 32768 \
#     --dynamic_sampling_policy confidence_gated \
#     --dynamic_sampling_kwargs '{"maxprob_threshold": 1.0, "T_high": 0.0}' \
#     --seed $seed \
#     --save_outputs ckpt/greedy_r1_distill_qwen7b_aime24_maj16_seed$seed.jsonl \
#     2>&1 | tee ckpt/aime24/greedy-r1-distill-qwen7b/maj16_seed$seed.log
# done