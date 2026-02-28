export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)

export WANDB_RUN_ID="AutoDeco-Thinking-$TS"

accelerate launch trl_train.py \
  --model_name_or_path ckpt/autodeco-qwen3-4b-thinking \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --output_dir ./autodeco_2heads_dolci \
  --train_temp true \
  --train_top_p false \
  --temp_objective legacy_ce \
  --easy_token_drop_prob 0.6 \
  --top_p_loss_method soft \
  --max_steps 3000 \
  --logging_steps 25 \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-6 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --temp_diag_enabled true \
  --temp_diag_steps 100 \
  --temp_diag_examples 3 \
  --temp_diag_topk 10 \
  --report_to wandb \
  --num_proc 16
