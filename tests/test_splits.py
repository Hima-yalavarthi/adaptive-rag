"""Leakage checks: train and test must not share example IDs."""

from src.ingestion.load_hotpotqa import load_hotpotqa_local


def test_train_and_validation_share_no_example_ids():
    train_ids = {ex["id"] for ex in load_hotpotqa_local("train")}
    val_ids = {ex["id"] for ex in load_hotpotqa_local("validation")}
    overlap = train_ids & val_ids
    assert overlap == set(), f"train/validation leak {len(overlap)} shared IDs, e.g. {next(iter(overlap))}"


def test_reported_eval_ids_are_validation_not_train():
    """The 100-question eval files must use validation IDs only."""
    from pathlib import Path
    import json

    train_ids = {ex["id"] for ex in load_hotpotqa_local("train")}
    val_ids = {ex["id"] for ex in load_hotpotqa_local("validation")}
    results = Path("evaluation/results")
    for name in ("baseline_n100_k5.jsonl", "reranked_n100_k5.jsonl", "verified_n100_k5.jsonl"):
        path = results / name
        if not path.exists():
            continue
        eval_ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
        assert eval_ids, f"{name} is empty"
        assert set(eval_ids) <= val_ids
        assert set(eval_ids).isdisjoint(train_ids)
