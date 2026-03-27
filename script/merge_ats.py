#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def copy_glob(src_dir: str, pattern: str, dst_dir: Path) -> int:
    matched = list(Path(src_dir).glob(pattern))
    for src_path in matched:
        if src_path.is_file():
            shutil.copy2(src_path, dst_dir / src_path.name)
    return len(matched)


def load_state_dicts(dir_path: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, load_file(os.path.join(dir_path, name)))
        for name in tqdm(os.listdir(dir_path), desc=f"Load {dir_path} safetensors")
        if name.endswith(".safetensors")
    ]


def merge_ats(
    ats_path: str,
    base_model_path: str,
    output_dir: str,
) -> None:
    logger.info("=" * 80)
    logger.info("ATS Model Merge")
    logger.info("=" * 80)
    base_state_dicts = load_state_dicts(base_model_path)
    index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_index = load_json(index_path)
    else:
        weight_index = {
            "metadata": {},
            "weight_map": {
                key: filename
                for filename, state_dict in base_state_dicts
                for key in state_dict.keys()
            },
        }
    merged_state_dicts = copy.deepcopy(base_state_dicts)
    merged_weight_index = copy.deepcopy(weight_index)
    for filename, state_dict in merged_state_dicts:
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("ats_head."):
                new_key = f"llm.{key}"
                new_state_dict[new_key] = value
                if key in merged_weight_index["weight_map"]:
                    merged_weight_index["weight_map"][new_key] = merged_weight_index["weight_map"].pop(key)
            else:
                new_state_dict[key] = value
        state_dict.clear()
        state_dict.update(new_state_dict)
    head_state_dict = {
        key: value
        for _, state_dict in load_state_dicts(ats_path)
        for key, value in state_dict.items()
        if key.startswith("ats_head.")
    }
    for key, value in head_state_dict.items():
        merged_state_dicts[-1][1][key] = value
        merged_weight_index["weight_map"][key] = merged_state_dicts[-1][0]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jinja_copied = copy_glob(base_model_path, "*.jinja", output_path)
    if jinja_copied == 0:
        copy_glob(ats_path, "*.jinja", output_path)
    copy_glob(base_model_path, "*.json", output_path)
    shutil.copy2(Path(ats_path) / "config.json", output_path / "config.json")
    write_json(os.path.join(output_dir, "model.safetensors.index.json"), merged_weight_index)
    for name, state_dict in merged_state_dicts:
        save_file(state_dict, os.path.join(output_dir, name))
    logger.info("Merged ATS checkpoint written to %s", output_dir)


def split_ats(
    full_checkpoint_path: str,
    output_dir: str,
) -> None:
    state_dicts = load_state_dicts(full_checkpoint_path)
    head_state_dict = {
        key: value
        for _, state_dict in state_dicts
        for key, value in state_dict.items()
        if key.startswith("ats_head.")
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    copy_glob(full_checkpoint_path, "*.jinja", output_path)
    copy_glob(full_checkpoint_path, "*.json", output_path)
    save_file(head_state_dict, os.path.join(output_dir, "ats_head.safetensors"))
    logger.info("Split ATS head checkpoint written to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ATS checkpoint merge/split utility")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--ats-path", type=str, required=True)
    merge_parser.add_argument("--base-model-path", type=str, required=True)
    merge_parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--full-checkpoint", type=str, required=True)
    split_parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "merge":
        merge_ats(args.ats_path, args.base_model_path, args.output_dir)
    else:
        split_ats(args.full_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
