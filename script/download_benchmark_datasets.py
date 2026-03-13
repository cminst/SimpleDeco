#!/usr/bin/env python3
"""Download several public benchmarks from Hugging Face and convert them to the
same JSONL schema as the user's AIME24 file:

    {"problem": <string>, "gt": <string>}

Benchmarks:
- AIME24            -> HuggingFaceH4/aime_2024 (train)
- AIME25            -> opencompass/AIME2025 (AIME2025-I + AIME2025-II, test)
- GPQA-Diamond      -> fingertap/GPQA-Diamond (test)
- MMLU-Pro          -> TIGER-Lab/MMLU-Pro (test)
- MMLU-Pro Lite     -> koiwave/100MMLUpro (train)
- BRUMO25           -> MathArena/brumo_2025 (train)
- HMMT25            -> MathArena/hmmt_feb_2025 (train)
- BeyondAIME        -> ByteDance-Seed/BeyondAIME (test)

For multiple-choice datasets, the output keeps the same two keys only:
- "problem": question text, with answer choices appended if they are stored separately
- "gt": by default, the correct option letter (A/B/C/...) for MCQ datasets

Examples:
    python build_same_format_jsonl.py
    python build_same_format_jsonl.py --outdir ./bench_jsonl
    python build_same_format_jsonl.py --mmlu-gt text
    python build_same_format_jsonl.py --gpqa-gt text
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from datasets import load_dataset

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _as_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def _clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _normalize_gt(value) -> str:
    gt = _clean(_as_str(value))
    if re.fullmatch(r"\d+", gt):
        gt = gt.lstrip("0")
        return gt or "0"
    return gt


def _looks_like_choices_already_present(question: str, min_labels: int = 4) -> bool:
    # Heuristic: many public MCQ datasets already embed "A. ... B. ... C. ..." in the question string.
    count = 0
    for label in LABELS[:10]:
        if re.search(rf"(?:^|\s){label}[\.)]\s", question):
            count += 1
    return count >= min_labels


def _append_choices(question: str, options: Sequence[str]) -> str:
    question = _clean(question)
    if _looks_like_choices_already_present(question, min_labels=min(4, len(options))):
        return question
    lines = [question, ""]
    for idx, opt in enumerate(options):
        label = LABELS[idx]
        lines.append(f"{label}. {_clean(_as_str(opt))}")
    return "\n".join(lines).strip()


def _option_letter_from_index(idx: int) -> str:
    if idx < 0 or idx >= len(LABELS):
        raise ValueError(f"answer_index out of range: {idx}")
    return LABELS[idx]


def _answer_text_from_letter(letter: str, options: Sequence[str]) -> str:
    letter = letter.strip().upper()
    pos = LABELS.index(letter)
    return _clean(_as_str(options[pos]))


def _write_jsonl(records: Iterable[Mapping[str, str]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_aime24() -> List[MutableMapping[str, str]]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        problem = row.get("problem") or row.get("question") or row.get("Problem")
        gt = row.get("answer") or row.get("gt") or row.get("Answer")
        if problem is None or gt is None:
            raise KeyError(f"Unexpected AIME24 row keys: {list(row.keys())}")
        out.append({"problem": _clean(_as_str(problem)), "gt": _normalize_gt(gt)})
    return out


def convert_aime25() -> List[MutableMapping[str, str]]:
    out: List[MutableMapping[str, str]] = []
    for config in ["AIME2025-I", "AIME2025-II"]:
        ds = load_dataset("opencompass/AIME2025", config, split="test")
        for row in ds:
            problem = row.get("problem") or row.get("question")
            gt = row.get("answer") or row.get("gt")
            if problem is None or gt is None:
                raise KeyError(f"Unexpected AIME25 row keys: {list(row.keys())}")
            out.append({"problem": _clean(_as_str(problem)), "gt": _normalize_gt(gt)})
    return out


def convert_gpqa(gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        question = row.get("question") or row.get("problem")
        if question is None:
            raise KeyError(f"Unexpected GPQA row keys: {list(row.keys())}")

        options = row.get("options")
        problem = _append_choices(_as_str(question), options) if options else _clean(_as_str(question))

        if gt_mode == "letter":
            gt = row.get("answer")
            if gt is None and row.get("answer_index") is not None:
                gt = _option_letter_from_index(int(row["answer_index"]))
            if gt is None:
                raise KeyError(f"Could not infer GPQA answer from row keys: {list(row.keys())}")
            gt = _clean(_as_str(gt))
        else:
            if options:
                if row.get("answer") is not None:
                    gt = _answer_text_from_letter(_as_str(row["answer"]), options)
                elif row.get("answer_index") is not None:
                    gt = _clean(_as_str(options[int(row["answer_index"])]))
                else:
                    raise KeyError(f"Could not infer GPQA answer text from row keys: {list(row.keys())}")
            else:
                # If the dataset only provides an answer letter and already-inlined choices, there is no reliable
                # generic parser here. Use the letter format instead.
                raise ValueError(
                    "GPQA text-mode ground truth requested, but this dataset instance does not expose separate options. "
                    "Use --gpqa-gt letter instead."
                )

        out.append({"problem": problem, "gt": _normalize_gt(gt)})
    return out


def _convert_mmlu_style(repo_id: str, split: str, gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    ds = load_dataset(repo_id, split=split)
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        question = row.get("question") or row.get("problem")
        options = row.get("options")
        if question is None or options is None:
            raise KeyError(f"Unexpected MMLU-Pro row keys: {list(row.keys())}")
        problem = _append_choices(_as_str(question), options)

        if gt_mode == "letter":
            if row.get("answer") is not None:
                gt = _clean(_as_str(row["answer"]))
            elif row.get("answer_index") is not None:
                gt = _option_letter_from_index(int(row["answer_index"]))
            else:
                raise KeyError(f"Could not infer {repo_id} answer from row keys: {list(row.keys())}")
        else:
            if row.get("answer_index") is not None:
                gt = _clean(_as_str(options[int(row["answer_index"])]))
            elif row.get("answer") is not None:
                gt = _answer_text_from_letter(_as_str(row["answer"]), options)
            else:
                raise KeyError(f"Could not infer {repo_id} answer text from row keys: {list(row.keys())}")

        out.append({"problem": problem, "gt": _normalize_gt(gt)})
    return out


def convert_mmlu_pro(gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    return _convert_mmlu_style("TIGER-Lab/MMLU-Pro", split="test", gt_mode=gt_mode)


def convert_mmlu_pro_lite(gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    return _convert_mmlu_style("koiwave/100MMLUpro", split="train", gt_mode=gt_mode)


def convert_matharena(repo_id: str, split: str = "train") -> List[MutableMapping[str, str]]:
    ds = load_dataset(repo_id, split=split)
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        problem = row.get("problem") or row.get("question")
        gt = row.get("answer") or row.get("gt")
        if problem is None or gt is None:
            raise KeyError(f"Unexpected {repo_id} row keys: {list(row.keys())}")
        out.append({"problem": _clean(_as_str(problem)), "gt": _normalize_gt(gt)})
    return out


def convert_beyondaime() -> List[MutableMapping[str, str]]:
    ds = load_dataset("ByteDance-Seed/BeyondAIME", split="test")
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        problem = row.get("problem") or row.get("question")
        gt = row.get("answer") or row.get("gt")
        if problem is None or gt is None:
            raise KeyError(f"Unexpected BeyondAIME row keys: {list(row.keys())}")
        out.append({"problem": _clean(_as_str(problem)), "gt": _normalize_gt(gt)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("same_format_jsonl"))
    parser.add_argument("--gpqa-gt", choices=["letter", "text"], default="letter")
    parser.add_argument("--mmlu-gt", choices=["letter", "text"], default="letter")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("aime24.jsonl", convert_aime24),
        ("aime25.jsonl", convert_aime25),
        ("gpqa_diamond.jsonl", lambda: convert_gpqa(gt_mode=args.gpqa_gt)),
        ("mmlu_pro.jsonl", lambda: convert_mmlu_pro(gt_mode=args.mmlu_gt)),
        ("mmlu_pro_lite.jsonl", lambda: convert_mmlu_pro_lite(gt_mode=args.mmlu_gt)),
        ("brumo25.jsonl", lambda: convert_matharena("MathArena/brumo_2025", split="train")),
        ("hmmt25.jsonl", lambda: convert_matharena("MathArena/hmmt_feb_2025", split="train")),
        ("beyondaime.jsonl", convert_beyondaime),
    ]

    summary = {}
    for filename, fn in jobs:
        records = fn()
        outpath = args.outdir / filename
        count = _write_jsonl(records, outpath)
        summary[filename] = {"rows": count, "path": str(outpath)}
        print(f"wrote {count:>6} rows -> {outpath}")

    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
