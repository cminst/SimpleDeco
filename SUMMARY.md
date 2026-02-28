# AutoDeco Repo Summary

## 1) Paper Summary (`paper.tex`)

The paper argues that current LLMs are not truly end-to-end because decoding hyperparameters (mainly temperature and top-p) are manually tuned and fixed at inference time.  

Its proposed solution, **AutoDeco**, adds lightweight heads on top of a frozen base LLM to predict **token-level temperature and top-p** from hidden states during generation. The key training challenge is that hard top-p is non-differentiable, so the paper introduces a **soft top-p surrogate** (exponential decay mask) to keep gradients flowing end-to-end.

Main claims in the paper:
- Better Pass@1 than greedy/default sampling across multiple model families.
- Close to oracle-style expert hyperparameter tuning without test-time manual sweeps.
- Minimal overhead (reported ~1-2% latency increase).
- Emergent natural-language control over decoding style (higher/lower diversity commands).

## 2) How This Repo Maps to the Paper

This codebase is a practical implementation of AutoDeco training and deployment, with the main training entrypoint at **`trl_train.py`**.

- `model/templlm_auto.py`
  - Wraps a base `AutoModelForCausalLM`.
  - Adds `TempHead` and `TopPHead` MLPs.
  - Implements:
    - temperature-head loss (`_compute_temp_loss`)
    - soft top-p loss (`_compute_top_p_loss`) matching the paper’s differentiable approximation idea.

- `trainer/trl_autodeco.py`
  - Custom `AutoDecoLLMTrainer` on top of TRL `SFTTrainer`.
  - Injects AutoDeco-specific loss settings into model forward.
  - Logs `loss`, `temp_loss`, `top_p_loss`.
  - Adds optional temperature diagnostics JSON dumping.

- `trl_train.py` (main)
  - Parses CLI args for training mode and objective.
  - Loads/normalizes dataset (including conversation-format checks).
  - Freezes/unfreezes base model vs heads depending on `--train_temp` / `--train_top_p`.
  - Launches either AutoDeco head training or standard SFT.

- Scripts/utilities
  - `script/construct_autodeco.py`: build AutoDeco wrapper from base model.
  - `script/merge_autodeco.py`: merge/split head weights for serving.
  - `utils/llm_eval.py`, `script/test_generation_*.sh`: evaluation pipeline.

## 3) Your Extension (from `conversation.md`)

You extended original AutoDeco with a **simplified temperature-first training path**:

- New objective: `analytic_min_p_hinge`
  - In code, temperature is trained with an analytic Min-P-inspired lower bound:
    - `required_temp = relu(max_logit - gt_logit) / (-log(min_p_ratio))`
    - loss includes hinge pressure toward this bound, plus optional regularization.
  - This enforces a “minimum sufficient temperature” safety condition instead of relying only on legacy CE behavior.

- Temp-only mode is first-class
  - `--train_temp true --train_top_p false`
  - Aligns with your “single-head simplification” idea.

- Goldilocks filtering added
  - Focuses training on selected easy/top-k tokens and ignores less useful regions.
  - Exposed via `--goldilocks_filter`, `--goldilocks_easy_frac`, `--goldilocks_topk`.

- Diagnostic tooling added
  - Optional per-step JSON dumps for predicted temp vs required temp and Min-P condition checks.
  - Makes the hinge objective behavior inspectable during training.

## 4) Current Status vs Paper Scope

- The repo now supports both:
  - original-style AutoDeco (temp + top-p heads),
  - your simplified/analytic temperature objective path.

- One important implementation note:
  - `model/templlm_auto.py` currently routes `generate()` to the base model directly.
  - So fully integrated dynamic decoding at generation time is expected to rely on the custom inference path (e.g., merge/runtime tooling), not the default HF `generate()` call in this wrapper.
