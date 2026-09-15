"""Load HotpotQA from curated JSON or the full local Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_FILES = {
    "train": DATA_DIR / "hotpotqa_train.json",
    "validation": DATA_DIR / "hotpotqa_validation.json",
}
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"


def load_hotpotqa_local(split: str = "validation") -> list[dict[str, Any]]:
    """Load a curated HotpotQA split from data/raw/*.json."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}. Choose from {tuple(SPLIT_FILES)}")

    path = SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected curated HotpotQA JSON in data/raw."
        )

    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{path} does not contain a non-empty list of examples.")

    return records


def load_hotpotqa_huggingface(split: str | None = None):
    """Load full HotpotQA distractor splits from the local Hugging Face cache."""
    from datasets import Dataset, DatasetDict, load_dataset

    load_dotenv(PROJECT_ROOT / ".env")
    cache_dir = Path(
        os.getenv("HOTPOTQA_CACHE_DIR", "data/raw/huggingface")
    ).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    return load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("local", "huggingface"),
        default="local",
        help="local = curated JSON in repo; huggingface = full HF cache (default: local).",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Which split(s) to load (default: all).",
    )
    args = parser.parse_args()

    if args.source == "local":
        splits = ("train", "validation") if args.split == "all" else (args.split,)
        total = 0
        print("Dataset: HotpotQA distractor (curated local JSON)")
        for name in splits:
            records = load_hotpotqa_local(name)
            total += len(records)
            example = records[0]
            print(f"\nSplit: {name}")
            print(f"Dataset size: {len(records):,} examples")
            print(f"File: {SPLIT_FILES[name].relative_to(PROJECT_ROOT)}")
            print(f"Example question: {example['question']}")
            print(f"Answer: {example['answer']}")
        print(f"\nTotal loaded: {total:,} examples")
        return

    from datasets import DatasetDict

    dataset = load_hotpotqa_huggingface(None if args.split == "all" else args.split)
    print(f"Dataset: {DATASET_ID} ({DATASET_CONFIG}) [huggingface cache]")
    splits = dataset if isinstance(dataset, DatasetDict) else {args.split: dataset}
    for name, records in splits.items():
        print(f"\nSplit: {name}")
        print(f"Dataset size: {len(records):,} examples")
        if len(records) == 0:
            raise RuntimeError(f"The HotpotQA {name} split contains no examples.")
        example = records[0]
        print(f"Example question: {example['question']}")
        print(f"Answer: {example['answer']}")
    print(
        f"\nTotal loaded: {sum(len(records) for records in splits.values()):,} examples"
    )


if __name__ == "__main__":
    main()
