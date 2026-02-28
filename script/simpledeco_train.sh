export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)

export WANDB_RUN_ID="ESAD-Thinking-$TS"

accelerate launch trl_train.py \
  --model_name_or_path ckpt/autodeco-qwen3-4b-thinking \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --output_dir ./run_reasoning_high_contrast \
  --train_temp true \
  --train_top_p false \
  --temp_objective analytic_min_p_hinge \
  --min_p_ratio 0.05 \
  --temp_hinge_weight 5.0 \
  --temp_reg_weight 1.1 \
  --temp_target_cap 0.8 \
  --goldilocks_filter false \
  --goldilocks_easy_frac 0.5 \
  --goldilocks_topk 15 \
  --max_steps 3000 \
  --logging_steps 25 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-4 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --temp_diag_enabled true \
  --temp_diag_steps 100 \
  --temp_diag_examples 3 \
  --temp_diag_topk 10 \
  --report_to wandb \
  --num_proc 16
