import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.templlm_auto import AutoDecoModelForCausalLM, AutoDecoModelForCausalLMConfig
from transformers import AutoConfig
import argparse
# Add project root (parent of this script directory) to sys.path for imports

def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def main(base_model_name_or_path, output_dir, enable_temperature_head, enable_top_p_head):
    from transformers import AutoTokenizer
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    autodeco_config = AutoDecoModelForCausalLMConfig(
        base_model_name_or_path=base_model_name_or_path,
        enable_temperature_head=enable_temperature_head,
        enable_top_p_head=enable_top_p_head,
        use_enhanced_features=True, # TODO: make it a parameter
        **base_config.to_dict()
    )
    model = AutoDecoModelForCausalLM(autodeco_config, dtype=base_config.dtype)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"AutoDeco model saved to {output_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--enable_temperature_head', type=parse_bool, default=True)
    parser.add_argument('--enable_top_p_head', type=parse_bool, default=True)
    args = parser.parse_args()

    main(
        args.base_model_name_or_path,
        args.output_dir,
        args.enable_temperature_head,
        args.enable_top_p_head,
    )
