"""Load curated HotpotQA JSON subsets from data/raw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_FILES = {
    "train": DATA_DIR / "hotpotqa_train.json",
    "validation": DATA_DIR / "hotpotqa_validation.json",
}


def load_hotpotqa(split: str = "validation") -> list[dict[str, Any]]:
    """Load a curated HotpotQA split saved as JSON in data/raw."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Which curated split(s) to load (default: all).",
    )
    args = parser.parse_args()

    splits = ("train", "validation") if args.split == "all" else (args.split,)
    total = 0
    print("Dataset: HotpotQA distractor (curated local JSON)")

    for name in splits:
        records = load_hotpotqa(name)
        total += len(records)
        example = records[0]
        print(f"\nSplit: {name}")
        print(f"Dataset size: {len(records):,} examples")
        print(f"File: {SPLIT_FILES[name].relative_to(PROJECT_ROOT)}")
        print(f"Example question: {example['question']}")
        print(f"Answer: {example['answer']}")

    print(f"\nTotal loaded: {total:,} examples")


if __name__ == "__main__":
    main()
