"""Chunk HotpotQA context passages into sentence-level retrieval units.

Each HotpotQA example includes titled paragraphs and sentence lists. This step
turns those sentences into chunks with metadata so later retrieval and
supporting-fact evaluation can reference title + sentence id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingestion.load_hotpotqa import PROJECT_ROOT, load_hotpotqa_local

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def context_to_chunks(
    example: dict[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Convert one HotpotQA example's context into sentence chunks."""
    example_id = example["id"]
    context = example["context"]
    titles = context["title"]
    sentence_lists = context["sentences"]

    if len(titles) != len(sentence_lists):
        raise ValueError(
            f"Example {example_id}: title/sentence list length mismatch "
            f"({len(titles)} vs {len(sentence_lists)})."
        )

    chunks: list[dict[str, Any]] = []
    for title, sentences in zip(titles, sentence_lists):
        for sent_id, sentence in enumerate(sentences):
            text = " ".join(str(sentence).split())
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{example_id}::{title}::{sent_id}",
                    "example_id": example_id,
                    "title": title,
                    "sent_id": sent_id,
                    "text": text,
                    "split": split,
                }
            )
    return chunks


def chunk_split(split: str) -> list[dict[str, Any]]:
    """Chunk every example in a curated local split."""
    examples = load_hotpotqa_local(split)
    chunks: list[dict[str, Any]] = []
    for example in examples:
        chunks.extend(context_to_chunks(example, split=split))
    return chunks


def save_chunks(chunks: list[dict[str, Any]], split: str) -> Path:
    """Write chunks as JSONL under data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"hotpotqa_{split}_chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="validation",
        help="Which curated split to chunk (default: validation).",
    )
    args = parser.parse_args()

    splits = ("train", "validation") if args.split == "all" else (args.split,)
    grand_total = 0

    for split in splits:
        chunks = chunk_split(split)
        path = save_chunks(chunks, split)
        grand_total += len(chunks)
        print(f"Split: {split}")
        print(f"Examples chunked: {len({c['example_id'] for c in chunks}):,}")
        print(f"Chunks written: {len(chunks):,}")
        print(f"Output: {path.relative_to(PROJECT_ROOT)}")
        if chunks:
            sample = chunks[0]
            print(f"Sample chunk_id: {sample['chunk_id']}")
            print(f"Sample text: {sample['text'][:160]}...")
        print()

    print(f"Total chunks: {grand_total:,}")


if __name__ == "__main__":
    main()
