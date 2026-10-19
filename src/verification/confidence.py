"""Combine verification and retrieval signals into a confidence score."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.retrieval.search import SearchHit
from src.verification.nli import VerificationResult

# Tuned so Verified RAG cuts confident errors without abstaining on most correct answers.
DEFAULT_THRESHOLD = 0.35


@dataclass
class ConfidenceResult:
    score: float
    threshold: float
    entailment: float
    retrieval: float
    citation_bonus: float
    accept: bool

    @property
    def reject(self) -> bool:
        return not self.accept


def _retrieval_strength(hits: list[SearchHit]) -> float:
    """Map top-hit scores into [0, 1]. Cross-encoder scores are unbounded; FAISS IP is in [-1, 1]."""
    if not hits:
        return 0.0
    top = float(hits[0].score)
    return float(1.0 / (1.0 + math.exp(-min(max(top, -20.0), 20.0))))


def score_confidence(
    verification: VerificationResult,
    hits: list[SearchHit],
    citations: list[int],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    w_support: float = 0.50,
    w_retrieval: float = 0.30,
    w_citation: float = 0.20,
) -> ConfidenceResult:
    """Weighted confidence in [0, 1].

    Support signal is (1 - contradiction) so neutral span answers are not
    punished as harshly as clear contradictions. Accept when the verifier
    marks the answer supported and the score clears the threshold.
    """
    retrieval = _retrieval_strength(hits)
    citation_bonus = 1.0 if citations else 0.0
    support = max(0.0, 1.0 - verification.contradiction)
    if verification.rival_entailment is not None:
        # yes/no: reward beating the opposite hypothesis
        support = max(support, max(0.0, verification.entailment - verification.rival_entailment))

    score = w_support * support + w_retrieval * retrieval + w_citation * citation_bonus
    score = max(0.0, min(1.0, score))
    accept = verification.supported and score >= threshold
    return ConfidenceResult(
        score=score,
        threshold=threshold,
        entailment=verification.entailment,
        retrieval=retrieval,
        citation_bonus=citation_bonus,
        accept=accept,
    )
