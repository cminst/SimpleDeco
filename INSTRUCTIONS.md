# AutoDeco TRL Run Instructions

## 1) Build AutoDeco wrapper from an instruct base

```bash
python script/construct_autodeco.py \
  --base_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir ckpt/autodeco-llama3-8b-instruct
```

Use the generated wrapper path (`ckpt/autodeco-llama3-8b-instruct`) as `--model_name_or_path` for training.

You might have to adjust the chat template to allow assistant masking (check `qwen3-4b-chat-template-fixed.jinja`)

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
  - `--goldilocks_easy_frac` (default `0.1`): target fraction of selected Goldilocks tokens where GT is top-1.
  - `--goldilocks_easy_frac -1`: disable balancing and keep the natural easy/top-k-non-easy ratio from the batch.
  - `--goldilocks_topk` (default `10`): the `k` for top-k membership.
- Filtering is strict to Goldilocks candidates only:
  - `easy`: GT rank = 1
  - `top-k non-easy`: GT rank in `[2, k]`
- Tokens outside top-k are not included by this filter.

## 5) Temperature diagnostics JSON dumps (during training)

Use these flags to periodically write debug JSON files with per-token temperature alignment details:

- `--temp_diag_enabled true`
- `--temp_diag_steps 100` (write every N global steps)
- `--temp_diag_examples 3` (max examples per file)
- `--temp_diag_topk 5` (top-k probabilities to include)
- `--temp_diag_dir temp_diagnostics` (subdir under `--output_dir`)

Example:

```bash
accelerate launch trl_train.py \
  ... \
  --temp_diag_enabled true \
  --temp_diag_steps 50 \
  --temp_diag_examples 3 \
  --temp_diag_topk 5
```

Output files:

- `<output_dir>/temp_diagnostics/step_0000050.json`
- `<output_dir>/temp_diagnostics/step_0000100.json`
- etc.

Each file includes up to 3 examples with:

- Context (`context_text`, `context_token_ids`)
- Ground-truth next token info (`token_id`, rank, probability)
- Predicted temperature (`prediction.predicted_temperature`)
- Required temperature from analytic Min-P bound (`min_p_alignment.required_temperature`)
- Hinge gap (`required - predicted`)
- Min-P condition check at predicted temperature
- Top token probabilities:
  - Unscaled logits
  - Logits scaled by predicted temperature

## 6) Pretty-print diagnostics helper

Use this helper to read JSON diagnostics and print a compact summary:

```bash
python script/pretty_print_temp_diag.py \
  --path ./debug_analytic_hinge/temp_diagnostics
```

Useful options:

- `--num-files 3` prints the latest 3 files from a directory.
- `--all` prints all `step_*.json` files in order.
- `--topk 10` shows more top-token entries.
- `--max-context-chars 600` prints longer context snippets.

Example:

```bash
python script/pretty_print_temp_diag.py \
  --path ./debug_analytic_hinge/temp_diagnostics \
  --num-files 2 \
  --topk 8
```
