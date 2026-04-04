export AUTODECO_NO_PROMPT=1
export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)

export WANDB_RUN_ID="AnalyticAlign-Qwen7B-$TS"

accelerate launch trl_train.py \
  --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --output_dir ./aa_r1_qwen7b_dolci \
  --train_temp true \
  --train_top_p false \
  --temp_objective taylor \
  --beta_bar 1.346 \
  --easy_token_drop_prob 0.0 \
  --goldilocks_uniform true \
  --goldilocks_uniform_bins 20 \
  --max_steps 3000 \
  --logging_steps 25 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --temp_diag_enabled true \
  --temp_diag_steps 100 \
  --temp_diag_examples 3 \
  --temp_diag_topk 10 \
  --dataset_train_split train_balanced \
  --report_to wandb \
  --num_proc 16
