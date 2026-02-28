export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)

export WANDB_RUN_ID="SimpleDeco-NewV2-$TS"

uv run accelerate launch trl_train.py \
  --model_name_or_path ckpt/untrained-autodeco-qwen3-4b-thinking \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --output_dir ./ckpt/simpledeco-bimodal \
  --train_temp true \
  --train_top_p false \
  --temp_objective analytic_min_p_hinge \
  --min_p_ratio 0.05 \
  --temp_hinge_weight 3.0 \
  --temp_reg_weight 1.5 \
  --temp_target_cap 0.9 \
  --temp_target_smooth_window 32 \
  --goldilocks_filter true \
  --goldilocks_easy_frac 0.4 \
  --goldilocks_topk 15 \
  --max_steps 3000 \
  --logging_steps 25 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --temp_diag_enabled true \
  --temp_diag_steps 100 \
  --temp_diag_examples 3 \
  --temp_diag_topk 10 \
  --report_to wandb \
  --num_proc 16
