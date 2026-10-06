import json

import faiss
import pytest

from src.retrieval.build_index import index_paths

SPLIT = "validation"
EXAMPLE_ID = "5a8b57f25542995d1e6f1371"
QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"


def _require_index():
    faiss_path, meta_path, config_path = index_paths(SPLIT)
    if not faiss_path.exists() or not meta_path.exists():
        pytest.skip("FAISS index not built; run python -m src.retrieval.build_index")
    return faiss_path, meta_path, config_path


def test_index_size_matches_metadata_and_config():
    faiss_path, meta_path, config_path = _require_index()
    index = faiss.read_index(str(faiss_path))
    with meta_path.open(encoding="utf-8") as f:
        num_meta = sum(1 for _ in f)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert index.ntotal == num_meta == config["num_vectors"]
    assert index.d == config["dimension"]


@pytest.fixture(scope="module")
def retriever():
    _require_index()
    from src.retrieval.search import FaissRetriever

    return FaissRetriever(split=SPLIT)


def test_search_returns_top_k_sorted_hits(retriever):
    hits = retriever.search(QUESTION, top_k=5)
    assert len(hits) == 5
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_example_filter_only_returns_that_examples_chunks(retriever):
    hits = retriever.search(QUESTION, top_k=5, example_id=EXAMPLE_ID)
    assert hits
    assert all(h.chunk["example_id"] == EXAMPLE_ID for h in hits)


def test_search_finds_gold_supporting_titles(retriever):
    hits = retriever.search(QUESTION, top_k=5, example_id=EXAMPLE_ID)
    titles = {h.chunk["title"] for h in hits}
    assert {"Scott Derrickson", "Ed Wood"} <= titles


def test_empty_query_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")
