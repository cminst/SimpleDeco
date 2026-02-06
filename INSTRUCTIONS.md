# AutoDeco TRL Run Instructions

## 1) Build AutoDeco wrapper from an instruct base

```bash
python script/construct_autodeco.py \
  --base_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir ckpt/autodeco-llama3-8b-instruct
```

Use the generated wrapper path (`ckpt/autodeco-llama3-8b-instruct`) as `--model_name_or_path` for training.

## 2) Run analytic Min-P hinge training

```bash
accelerate launch trl_train.py \
  --model_name_or_path ckpt/autodeco-llama3-8b-instruct \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --output_dir ./debug_analytic_hinge \
  --train_temp true \
  --train_top_p false \
  --temp_objective analytic_min_p_hinge \
  --min_p_ratio 0.1 \
  --temp_hinge_weight 1.0 \
  --temp_reg_weight 0.1 \
  --goldilocks_filter true \
  --goldilocks_easy_frac 0.1 \
  --goldilocks_topk_frac 0.9 \
  --goldilocks_topk 10 \
  --max_steps 50 \
  --logging_steps 1 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-4 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --gradient_checkpointing \
  --completion_only_loss true \
  --report_to wandb
```

## 3) Dataset behavior in `trl_train.py`

- Local JSON/JSONL also works: pass a file name under `data/` (for example `deepmath_30k_trl.json`) or a direct file path.
- HF dataset IDs also work (for example `HuggingFaceH4/ultrachat_200k`).
- The script prompts for train split selection.
- If the selected split has a `messages` or `conversations` column, it auto-selects that column.
- Otherwise, the script prompts you to choose a column.
- For `messages`/`conversations`, it runs a sanity check on sampled rows to verify data roughly matches:
  - list of dict turns
  - each turn has `role` and `content` string fields

If the sanity check finds too many malformed rows, training stops with an error.

## 4) Goldilocks filtering (analytic Min-P objective)

- `--goldilocks_filter true` enables token-level filtering for `analytic_min_p_hinge`.
- Token mix target is controlled by:
  - `--goldilocks_easy_frac` (default `0.1`): fraction of tokens where GT is top-1.
  - `--goldilocks_topk_frac` (default `0.9`): fraction of tokens where GT is in top-k.
  - `--goldilocks_topk` (default `10`): the `k` for top-k membership.
- If a batch does not have enough tokens in one bucket, the remainder is filled from available top-k tokens, then any remaining valid tokens.
