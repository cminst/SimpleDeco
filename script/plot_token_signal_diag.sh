python3 script/plot_token_signal_diagnostics.py \
  --model_name_or_path ckpt/simpledeco-qwen3-4b-thinking-merged \
  --dataset_name qingy2024/Dolci-Think-SFT-ctx8k \
  --dataset_split val_balanced \
  --dataset_text_field messages \
  --assistant_only \
  --batch_size 2