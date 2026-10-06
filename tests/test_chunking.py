import json

import pytest

from src.ingestion.chunk_hotpotqa import PROCESSED_DIR, context_to_chunks

EXAMPLE = {
    "id": "ex1",
    "context": {
        "title": ["Ed Wood", "Scott Derrickson"],
        "sentences": [
            ["Ed Wood was an American filmmaker.", "   ", " He made  low-budget films. "],
            ["Scott Derrickson is an American director."],
        ],
    },
}

REQUIRED_KEYS = {"chunk_id", "example_id", "title", "sent_id", "text", "split"}


def test_one_chunk_per_non_empty_sentence():
    chunks = context_to_chunks(EXAMPLE, split="validation")
    assert len(chunks) == 3


def test_chunk_fields_and_id_format():
    chunk = context_to_chunks(EXAMPLE, split="validation")[0]
    assert set(chunk) == REQUIRED_KEYS
    assert chunk["chunk_id"] == "ex1::Ed Wood::0"
    assert chunk["split"] == "validation"


def test_sent_id_matches_original_position_after_skipping_blank():
    chunks = context_to_chunks(EXAMPLE, split="validation")
    ed_wood = [c for c in chunks if c["title"] == "Ed Wood"]
    assert [c["sent_id"] for c in ed_wood] == [0, 2]


def test_whitespace_is_normalized():
    chunks = context_to_chunks(EXAMPLE, split="validation")
    assert chunks[1]["text"] == "He made low-budget films."


def test_mismatched_titles_and_sentences_raise():
    bad = {"id": "bad", "context": {"title": ["A", "B"], "sentences": [["x"]]}}
    with pytest.raises(ValueError):
        context_to_chunks(bad, split="train")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_committed_chunk_files_are_well_formed(split):
    path = PROCESSED_DIR / f"hotpotqa_{split}_chunks.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not generated")

    seen_ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            assert set(chunk) == REQUIRED_KEYS
            assert chunk["text"].strip()
            assert chunk["split"] == split
            assert chunk["chunk_id"] not in seen_ids
            seen_ids.add(chunk["chunk_id"])
    assert seen_ids
