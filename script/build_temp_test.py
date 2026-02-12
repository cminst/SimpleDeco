import argparse
import json
import os

from datasets import load_dataset


def build_aime24(out_path: str) -> None:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({"problem": row["problem"], "gt": row["answer"]}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aime24")
    parser.add_argument("--out", type=str, default="data/TempTest/aime24.jsonl")
    args = parser.parse_args()

    if args.dataset != "aime24":
        raise ValueError(f"Unsupported dataset: {args.dataset}. Only 'aime24' is supported for now.")

    build_aime24(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
