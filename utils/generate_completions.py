"""
Generate completions from an OpenAI-compatible API endpoint and upload as an HF dataset.

Usage:
    python utils/generate_completions.py \
        --base-url http://localhost:8000/v1 \
        --model my-model \
        --max-tokens 2048 \
        --dataset lmsys/lmsys-chat-1m \
        --n 1000 \
        --repo-id myuser/my-completions-dataset \
        [--api-key sk-...] \
        [--dataset-config default] \
        [--dataset-split train] \
        [--concurrency 64] \
        [--temperature 1.0] \
        [--top-p 1.0] \
        [--private]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm


def _load_prompts(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str | None,
    n: int,
) -> list[list[dict[str, Any]]]:
    """Load the first n prompts (first user message) from the dataset."""
    kwargs: dict[str, Any] = {}
    if dataset_config:
        kwargs["name"] = dataset_config

    # Support local JSON/JSONL files
    if os.path.isfile(dataset_name) or os.path.isfile(os.path.join("data", dataset_name)):
        path = dataset_name if os.path.isfile(dataset_name) else os.path.join("data", dataset_name)
        ds = load_dataset("json", data_files=path)
    else:
        ds = load_dataset(dataset_name, **kwargs)

    # Resolve split
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        split = dataset_split if dataset_split in splits else splits[0]
        ds = ds[split]

    if "messages" not in ds.column_names:
        raise ValueError(f"Dataset must have a 'messages' column; found: {ds.column_names}")

    prompts = []
    for row in ds:
        messages: list[dict] = row["messages"]
        # Keep only the first user turn (strip trailing assistant if present)
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user is None:
            continue
        prompts.append([first_user])
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        print(f"Warning: only {len(prompts)} prompts available (requested {n}).", file=sys.stderr)

    return prompts


async def _complete_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str | None:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            return None


async def _run_batch(
    base_url: str,
    api_key: str,
    model: str,
    prompts: list[list[dict]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
) -> list[str | None]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _complete_one(client, sem, model, msgs, max_tokens, temperature, top_p)
        for msgs in prompts
    ]
    results = await atqdm.gather(*tasks, desc="Generating completions")
    await client.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate completions and upload as HF dataset")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var, or 'none')")
    parser.add_argument("--model", required=True, help="Model name as known to the endpoint")
    parser.add_argument("--max-tokens", type=int, required=True, help="Max tokens to generate per response")
    parser.add_argument("--dataset", required=True, help="HF dataset name or local JSON/JSONL file path")
    parser.add_argument("--n", type=int, required=True, help="Number of prompts to run")
    parser.add_argument("--repo-id", required=True, help="HF repo to push the output dataset to (e.g. myuser/my-ds)")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--concurrency", type=int, default=64, help="Max concurrent API requests (default: 64)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--private", action="store_true", help="Make the uploaded HF dataset private")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "none"

    print(f"Loading {args.n} prompts from '{args.dataset}'...")
    prompts = _load_prompts(args.dataset, args.dataset_config, args.dataset_split, args.n)
    print(f"Loaded {len(prompts)} prompts.")

    print(f"Running completions against {args.base_url} (model={args.model}, concurrency={args.concurrency})...")
    completions = asyncio.run(_run_batch(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
    ))

    # Build dataset: skip rows where the completion failed
    rows = []
    skipped = 0
    for prompt_msgs, completion in zip(prompts, completions):
        if completion is None:
            skipped += 1
            continue
        rows.append({"messages": [prompt_msgs[0], {"role": "assistant", "content": completion}]})

    if skipped:
        print(f"Warning: {skipped} completions failed and were skipped.", file=sys.stderr)

    print(f"Building dataset with {len(rows)} rows...")
    ds = Dataset.from_list(rows)

    print(f"Pushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
