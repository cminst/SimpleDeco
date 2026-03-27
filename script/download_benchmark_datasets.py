"""Download several public benchmarks from Hugging Face and convert them to the
following JSONL schema:

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
- IFEval            -> google/IFEval (train)
- IFBench           -> allenai/IFBench_test (train)

For multiple-choice datasets, the output keeps the same two keys only:
- "problem": question text, with answer choices appended if they are stored separately
- "gt": by default, the correct option letter (A/B/C/...) for MCQ datasets

Examples:
    python build_same_format_jsonl.py
    python build_same_format_jsonl.py --output-dir data/TempTest
    python build_same_format_jsonl.py --mmlu-gt text
    python build_same_format_jsonl.py --gpqa-gt text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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


def _normalize_gt(answer: str) -> str:
    answer = _clean(answer)
    if re.fullmatch(r"\d+", answer):
        answer = answer.lstrip("0")
        return answer or "0"
    return answer


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
            rec_to_write = dict(rec)
            rec_to_write["gt"] = _normalize_gt(_as_str(rec_to_write.get("gt")))
            f.write(json.dumps(rec_to_write, ensure_ascii=False) + "\n")
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
        out.append({"problem": _clean(_as_str(problem)), "gt": _clean(_as_str(gt))})
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
            out.append({"problem": _clean(_as_str(problem)), "gt": _clean(_as_str(gt))})
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

        out.append({"problem": problem, "gt": gt})
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

        out.append({"problem": problem, "gt": gt})
    return out


def convert_mmlu_pro(gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    return _convert_mmlu_style("TIGER-Lab/MMLU-Pro", split="test", gt_mode=gt_mode)


def convert_mmlu_pro_lite(gt_mode: str = "letter") -> List[MutableMapping[str, str]]:
    return _convert_mmlu_style("koiwave/100MMLUpro", split="train", gt_mode=gt_mode)


def _normalize_category(category: str) -> str:
    cat = _clean(_as_str(category)).lower()
    aliases = {
        "others": "other",
        "computer science": "computer science",
    }
    return aliases.get(cat, cat)


def _mmlu_fingerprint(row: Mapping[str, object]) -> str:
    question = _clean(_as_str(row.get("question") or row.get("problem")))
    options = row.get("options") or []
    options_text = "\n".join(_clean(_as_str(x)) for x in options)
    category = _normalize_category(_as_str(row.get("category")))
    answer = _clean(_as_str(row.get("answer")))
    src = _clean(_as_str(row.get("src")))
    return "\u241f".join([question, options_text, category, answer, src])


def _stable_sort_key(seed: int, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}::{fingerprint}".encode("utf-8")).hexdigest()


def _allocate_proportional_counts(counts: Mapping[str, int], total: int) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    total_available = sum(int(v) for v in counts.values())
    if total_available <= 0:
        raise ValueError("counts must sum to a positive number")

    raw = {k: (float(v) / total_available) * total for k, v in counts.items()}
    alloc = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = total - sum(alloc.values())

    # Largest-remainder apportionment to hit the requested total exactly.
    # Ties are broken deterministically by category name.
    order = sorted(raw.keys(), key=lambda k: (-(raw[k] - alloc[k]), k))
    for k in order[:remaining]:
        alloc[k] += 1
    return alloc


def convert_general_dev(mmlu_gt_mode: str = "letter", seed: int = 1337) -> List[MutableMapping[str, str]]:
    """Build a 60-question mixed dev set:

    - 30 questions from BRUMO25
    - 30 questions from TIGER-Lab/MMLU-Pro test, excluding any question that appears
      in koiwave/100MMLUpro, with category proportions matched to the *observed*
      koiwave/100MMLUpro category histogram.

    This mirrors the lite subset's effective stratification while avoiding the minor
    inconsistencies in the dataset card's written quota table.
    """
    full_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    lite_ds = load_dataset("koiwave/100MMLUpro", split="train")

    lite_fingerprints = {_mmlu_fingerprint(row) for row in lite_ds}
    lite_category_counts: Dict[str, int] = {}
    for row in lite_ds:
        cat = _normalize_category(_as_str(row.get("category")))
        lite_category_counts[cat] = lite_category_counts.get(cat, 0) + 1

    remaining_by_category: Dict[str, List[Mapping[str, object]]] = {}
    for row in full_ds:
        fp = _mmlu_fingerprint(row)
        if fp in lite_fingerprints:
            continue
        cat = _normalize_category(_as_str(row.get("category")))
        remaining_by_category.setdefault(cat, []).append(row)

    target_counts = _allocate_proportional_counts(lite_category_counts, total=30)

    # Safety: if a category is exhausted after exclusion, reallocate shortfall by the
    # same largest-remainder logic over categories with spare capacity.
    sampled_rows: List[Tuple[str, Mapping[str, object]]] = []
    deficits = 0
    spare_capacities: Dict[str, int] = {}

    for cat, target in target_counts.items():
        candidates = remaining_by_category.get(cat, [])
        ordered = sorted(candidates, key=lambda r: _stable_sort_key(seed, _mmlu_fingerprint(r)))
        take = min(target, len(ordered))
        sampled_rows.extend((cat, row) for row in ordered[:take])
        deficits += target - take
        spare_capacities[cat] = max(0, len(ordered) - take)

    if deficits > 0:
        spare_order = sorted(
            [cat for cat, spare in spare_capacities.items() if spare > 0],
            key=lambda cat: (-(lite_category_counts.get(cat, 0)), cat),
        )
        for cat in spare_order:
            if deficits <= 0:
                break
            candidates = remaining_by_category.get(cat, [])
            ordered = sorted(candidates, key=lambda r: _stable_sort_key(seed, _mmlu_fingerprint(r)))
            already_taken = sum(1 for taken_cat, _ in sampled_rows if taken_cat == cat)
            available_extra = len(ordered) - already_taken
            extra_take = min(deficits, available_extra)
            if extra_take > 0:
                sampled_rows.extend((cat, row) for row in ordered[already_taken:already_taken + extra_take])
                deficits -= extra_take

    if deficits != 0:
        raise RuntimeError(f"Could not allocate 30 MMLU-Pro dev questions after excluding lite subset; short by {deficits}.")

    # Drop category labels and convert to the same JSONL schema.
    mmlu_rows = [row for _, row in sampled_rows]
    # Keep deterministic overall ordering by category then hashed fingerprint.
    mmlu_rows = sorted(
        mmlu_rows,
        key=lambda r: (
            _normalize_category(_as_str(r.get("category"))),
            _stable_sort_key(seed, _mmlu_fingerprint(r)),
        ),
    )

    mmlu_out: List[MutableMapping[str, str]] = []
    for row in mmlu_rows:
        question = row.get("question") or row.get("problem")
        options = row.get("options")
        if question is None or options is None:
            raise KeyError(f"Unexpected MMLU-Pro row keys in general_dev: {list(row.keys())}")
        problem = _append_choices(_as_str(question), options)
        if mmlu_gt_mode == "letter":
            if row.get("answer") is not None:
                gt = _clean(_as_str(row["answer"]))
            elif row.get("answer_index") is not None:
                gt = _option_letter_from_index(int(row["answer_index"]))
            else:
                raise KeyError(f"Could not infer MMLU-Pro answer from row keys: {list(row.keys())}")
        else:
            if row.get("answer_index") is not None:
                gt = _clean(_as_str(options[int(row["answer_index"])]))
            elif row.get("answer") is not None:
                gt = _answer_text_from_letter(_as_str(row["answer"]), options)
            else:
                raise KeyError(f"Could not infer MMLU-Pro answer text from row keys: {list(row.keys())}")
        mmlu_out.append({"problem": problem, "gt": gt})

    brumo_out = convert_matharena("MathArena/brumo_2025", split="train")
    if len(brumo_out) != 30:
        raise RuntimeError(f"Expected 30 BRUMO25 rows, got {len(brumo_out)}")

    return mmlu_out + brumo_out


def convert_matharena(repo_id: str, split: str = "train") -> List[MutableMapping[str, str]]:
    ds = load_dataset(repo_id, split=split)
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        problem = row.get("problem") or row.get("question")
        gt = row.get("answer") or row.get("gt")
        if problem is None or gt is None:
            raise KeyError(f"Unexpected {repo_id} row keys: {list(row.keys())}")
        out.append({"problem": _clean(_as_str(problem)), "gt": _clean(_as_str(gt))})
    return out


def convert_beyondaime() -> List[MutableMapping[str, str]]:
    ds = load_dataset("ByteDance-Seed/BeyondAIME", split="test")
    out: List[MutableMapping[str, str]] = []
    for row in ds:
        problem = row.get("problem") or row.get("question")
        gt = row.get("answer") or row.get("gt")
        if problem is None or gt is None:
            raise KeyError(f"Unexpected BeyondAIME row keys: {list(row.keys())}")
        out.append({"problem": _clean(_as_str(problem)), "gt": _clean(_as_str(gt))})
    return out


def _write_raw_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> int:
    """Write records as-is (no gt normalization). Used for instruction-following datasets."""
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_ifeval() -> List[Mapping[str, object]]:
    """Download google/IFEval and return raw records with prompt/instruction_id_list/kwargs/key."""
    ds = load_dataset("google/IFEval", split="train")
    out: List[Mapping[str, object]] = []
    for row in ds:
        out.append({
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })
    return out


def convert_ifbench() -> List[Mapping[str, object]]:
    """Download allenai/IFBench_test and return raw records."""
    ds = load_dataset("allenai/IFBench_test", split="train")
    out: List[Mapping[str, object]] = []
    for row in ds:
        out.append({
            "key": row["key"],
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=Path("same_format_jsonl"))
    parser.add_argument("--gpqa-gt", choices=["letter", "text"], default="letter")
    parser.add_argument("--mmlu-gt", choices=["letter", "text"], default="letter")
    parser.add_argument("--general-dev-seed", type=int, default=1337)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("aime24.jsonl", convert_aime24),
        ("aime25.jsonl", convert_aime25),
        ("gpqa_diamond.jsonl", lambda: convert_gpqa(gt_mode=args.gpqa_gt)),
        ("mmlu_pro.jsonl", lambda: convert_mmlu_pro(gt_mode=args.mmlu_gt)),
        ("mmlu_pro_lite.jsonl", lambda: convert_mmlu_pro_lite(gt_mode=args.mmlu_gt)),
        ("general_dev.jsonl", lambda: convert_general_dev(mmlu_gt_mode=args.mmlu_gt, seed=args.general_dev_seed)),
        ("brumo25.jsonl", lambda: convert_matharena("MathArena/brumo_2025", split="train")),
        ("hmmt25.jsonl", lambda: convert_matharena("MathArena/hmmt_feb_2025", split="train")),
        ("beyondaime.jsonl", convert_beyondaime),
    ]

    # Instruction-following datasets use a different schema (no gt normalization).
    if_jobs = [
        ("ifeval.jsonl", convert_ifeval),
        ("ifbench.jsonl", convert_ifbench),
    ]

    summary = {}
    for filename, fn in jobs:
        records = fn()
        outpath = args.output_dir / filename
        count = _write_jsonl(records, outpath)
        summary[filename] = {"rows": count, "path": str(outpath)}
        print(f"wrote {count:>6} rows -> {outpath}")

    for filename, fn in if_jobs:
        records = fn()
        outpath = args.output_dir / filename
        count = _write_raw_jsonl(records, outpath)
        summary[filename] = {"rows": count, "path": str(outpath)}
        print(f"wrote {count:>6} rows -> {outpath}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
