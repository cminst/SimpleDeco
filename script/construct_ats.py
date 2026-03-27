#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ATSConfig, ATSModelForCausalLM
from transformers import AutoConfig, AutoTokenizer


def main(
    base_model_name_or_path: str,
    output_dir: str,
    calibration_type: str,
    feature_key: str,
    normalize_logits: bool,
    max_temperature: float,
    loss_type: str,
    label_smoothing: float,
    smooth_loss_weight: float,
    label_smoothing_type: str,
    smoothing_topk: int,
) -> None:
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    ats_config = ATSConfig(
        base_model_name_or_path=base_model_name_or_path,
        base_model_type=getattr(base_config, "model_type", None),
        calibration_type=calibration_type,
        feature_key=feature_key,
        normalize_logits=normalize_logits,
        max_temperature=max_temperature,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        smooth_loss_weight=smooth_loss_weight,
        label_smoothing_type=label_smoothing_type,
        smoothing_topk=smoothing_topk,
        **base_config.to_dict(),
    )
    base_model_kwargs = {}
    torch_dtype = getattr(base_config, "torch_dtype", None)
    if torch_dtype is not None:
        base_model_kwargs["torch_dtype"] = torch_dtype
    model = ATSModelForCausalLM(ats_config, base_model_kwargs=base_model_kwargs)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"ATS model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--calibration_type", type=str, default="transformer")
    parser.add_argument("--feature_key", type=str, default="hidden_states")
    parser.add_argument("--normalize_logits", action="store_true")
    parser.add_argument("--max_temperature", type=float, default=10.0)
    parser.add_argument("--loss_type", type=str, default="selective_smoothing")
    parser.add_argument("--label_smoothing", type=float, default=1.0)
    parser.add_argument("--smooth_loss_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing_type", type=str, default="uniform")
    parser.add_argument("--smoothing_topk", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))
