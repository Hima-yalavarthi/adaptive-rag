from src.reranking.cross_encoder import CrossEncoderReranker, rerank_text
from src.retrieval.search import SearchHit


class FakeScorer:
    """Scores a pair by how many query words appear in the passage."""

    def __init__(self):
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return [
            len(set(query.lower().split()) & set(passage.lower().replace(":", "").split()))
            for query, passage in pairs
        ]


def hit(title, text, score):
    return SearchHit(score=score, chunk={"title": title, "sent_id": 0, "text": text})


HITS = [
    hit("A", "unrelated sentence", 0.9),
    hit("B", "ed wood was american", 0.5),
    hit("C", "wood carving", 0.7),
]


def test_rerank_orders_by_cross_encoder_score_and_truncates():
    reranker = CrossEncoderReranker(model=FakeScorer())
    ranked = reranker.rerank("ed wood american", HITS, top_k=2)
    assert [h.chunk["title"] for h in ranked] == ["B", "C"]
    assert [h.score for h in ranked] == [3.0, 1.0]


def test_rerank_pairs_include_title():
    scorer = FakeScorer()
    CrossEncoderReranker(model=scorer).rerank("q", HITS[:1], top_k=1)
    assert scorer.pairs == [("q", "A: unrelated sentence")]
    assert rerank_text(HITS[1].chunk) == "B: ed wood was american"


def test_rerank_empty_hits():
    assert CrossEncoderReranker(model=FakeScorer()).rerank("q", [], top_k=5) == []
