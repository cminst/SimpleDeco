export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)

export WANDB_RUN_ID="ESAD-enforced-AOM-fixed-$TS"


accelerate launch trl_train.py \
  --model_name_or_path ckpt/autodeco-qwen3-4b-instruct \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --output_dir ./debug_analytic_hinge \
  --train_temp true \
  --train_top_p false \
  --temp_objective analytic_min_p_hinge \
  --min_p_ratio 0.05 \
  --temp_hinge_weight 1.0 \
  --temp_reg_weight 0.2 \
  --goldilocks_filter true \
  --goldilocks_easy_frac 0.4 \
  --goldilocks_topk_frac 0.6 \
  --goldilocks_topk 10 \
  --max_steps 1000 \
  --logging_steps 25 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --temp_diag_enabled true \
  --temp_diag_steps 100 \
  --temp_diag_examples 3 \
  --temp_diag_topk 10 \
  --report_to wandb
