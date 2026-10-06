"""Rerank retrieved sentences with a cross-encoder.

A bi-encoder (the FAISS retriever) embeds the question and each sentence
separately. A cross-encoder reads them together, which is slower but usually
ranks relevance more accurately, so it is applied only to a small candidate set.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.retrieval.search import SearchHit

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


def rerank_text(chunk: dict[str, Any]) -> str:
    return f"{chunk['title']}: {chunk['text']}"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, model: PairScorer | None = None):
        self.model_name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            try:
                model = CrossEncoder(model_name, local_files_only=True)
            except (OSError, ValueError, EnvironmentError):
                model = CrossEncoder(model_name)
        self.model = model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return the top_k hits ordered by cross-encoder score (scores replace FAISS scores)."""
        if not hits:
            return []
        pairs = [(query, rerank_text(h.chunk)) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, hits), key=lambda pair: float(pair[0]), reverse=True)
        return [SearchHit(score=float(score), chunk=hit.chunk) for score, hit in ranked[:top_k]]
