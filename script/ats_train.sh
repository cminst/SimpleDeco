export WANDB_PROJECT="ESAD"
TS=$(date +%Y%m%d%H%M%S)
export WANDB_RUN_ID="ATS-Dolci-$TS"

accelerate launch trl_train.py \
  --model_name_or_path ckpt/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --output-dir ./ats_dolci \
  --train_ats true \
  --ats_calibration_type transformer \
  --ats_feature_key hidden_states \
  --ats_loss_type selective_smoothing \
  --ats_label_smoothing 0.3 \
  --ats_smooth_loss_weight 0.5 \
  --ats_label_smoothing_type uniform \
  --ats_max_temperature 10.0 \
  --max_steps 3000 \
  --logging_steps 25 \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-6 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --assistant_only_loss true \
  --report_to wandb \
  --num_proc 16
