"""Load HotpotQA distractor splits and confirm dataset access."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"


def load_hotpotqa(split: str | None = None) -> Dataset | DatasetDict:
    """Load all distractor splits by default, or one selected split."""
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
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Which split(s) to load (default: all).",
    )
    args = parser.parse_args()

    dataset = load_hotpotqa(None if args.split == "all" else args.split)
    print(f"Dataset: {DATASET_ID} ({DATASET_CONFIG})")

    splits = dataset if isinstance(dataset, DatasetDict) else {args.split: dataset}
    for name, records in splits.items():
        print(f"\nSplit: {name}")
        print(f"Dataset size: {len(records):,} examples")
        if len(records) == 0:
            raise RuntimeError(f"The HotpotQA {name} split contains no examples.")
        example = records[0]
        print(f"Example question: {example['question']}")
        print(f"Answer: {example['answer']}")

    total = sum(len(records) for records in splits.values())
    print(f"\nTotal loaded: {total:,} examples")


if __name__ == "__main__":
    main()
