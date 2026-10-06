from src.generation.llm import GeneratedAnswer
from src.pipeline.baseline import BaselineRAG, gold_facts
from src.retrieval.search import SearchHit

CHUNKS = [
    {"title": "Scott Derrickson", "sent_id": 0, "text": "Scott Derrickson is an American director."},
    {"title": "Ed Wood", "sent_id": 0, "text": "Ed Wood was an American filmmaker."},
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=5, example_id=None):
        self.calls.append((query, top_k, example_id))
        return [SearchHit(score=1.0 - i / 10, chunk=c) for i, c in enumerate(CHUNKS[:top_k])]


class FakeGenerator:
    def __init__(self):
        self.received = None

    def generate(self, question, chunks):
        self.received = chunks
        return GeneratedAnswer(answer="yes", citations=[1, 2], cited_chunks=chunks, latency_s=0.5)


def test_baseline_passes_retrieved_chunks_to_generator():
    retriever, generator = FakeRetriever(), FakeGenerator()
    result = BaselineRAG(retriever, generator, top_k=2).answer("Same nationality?", example_id="ex1")

    assert retriever.calls == [("Same nationality?", 2, "ex1")]
    assert generator.received == CHUNKS
    assert result.answer == "yes"
    assert result.citations == [1, 2]
    assert len(result.evidence) == 2
    assert result.total_s >= result.generation_s == 0.5


class ReverseReranker:
    def rerank(self, query, hits, top_k):
        return list(reversed(hits))[:top_k]


def test_reranked_pipeline_fetches_candidates_then_reranks():
    retriever, generator = FakeRetriever(), FakeGenerator()
    rag = BaselineRAG(retriever, generator, top_k=1, reranker=ReverseReranker(), candidate_k=2)
    result = rag.answer("Same nationality?")

    assert retriever.calls == [("Same nationality?", 2, None)]
    assert generator.received == [CHUNKS[1]]
    assert [h.chunk for h in result.evidence] == [CHUNKS[1]]
    assert result.rerank_s >= 0


def test_gold_facts_pairs_titles_with_sentence_ids():
    example = {"supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]}}
    assert gold_facts(example) == {("Scott Derrickson", 0), ("Ed Wood", 0)}
