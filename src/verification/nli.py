"""Evidence verification with a cross-encoder NLI model.

Given a question, a generated answer, and retrieved sentences, score whether
the evidence entails the answer. Used to catch confidently wrong generations
before returning them to the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

DEFAULT_NLI = "cross-encoder/nli-deberta-v3-xsmall"

# sentence-transformers NLI cross-encoders use this label order.
LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT = 1
LABEL_NEUTRAL = 2


class TripletScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class VerificationResult:
    supported: bool
    entailment: float
    contradiction: float
    neutral: float
    premise: str
    hypothesis: str
    used_citations: bool
    latency_s: float = 0.0
    rival_entailment: float | None = None  # for yes/no: entailment of the opposite answer

    @property
    def support_score(self) -> float:
        """Entailment mass, the main support signal for confidence scoring."""
        return self.entailment


def evidence_text(chunks: list[dict[str, Any]], citations: list[int] | None = None) -> tuple[str, bool]:
    """Build the NLI premise from cited sentences, or all evidence if none cited."""
    if citations:
        selected = [chunks[i - 1] for i in citations if 1 <= i <= len(chunks)]
        if selected:
            text = " ".join(f"({c['title']}) {c['text']}" for c in selected)
            return text, True
    text = " ".join(f"({c['title']}) {c['text']}" for c in chunks)
    return text, False


def _as_statement(question: str) -> str:
    q = question.strip().rstrip("?")
    return q


def _negated_statement(question: str) -> str:
    q = _as_statement(question)
    if not q:
        return "No."
    rest = q[0].lower() + q[1:] if len(q) > 1 else q.lower()
    return f"It is not the case that {rest}."


def build_hypothesis(question: str, answer: str) -> str:
    """Turn a short HotpotQA-style answer into an NLI hypothesis."""
    ans = answer.strip()
    lower = ans.lower()
    if lower == "yes":
        return _as_statement(question)
    if lower == "no":
        return _negated_statement(question)
    return f"The answer to the question is {ans}."


def _softmax_probs(logits: np.ndarray) -> tuple[float, float, float]:
    row = np.asarray(logits, dtype=float).reshape(-1)
    exp = np.exp(row - np.max(row))
    probs = exp / exp.sum()
    return float(probs[LABEL_CONTRADICTION]), float(probs[LABEL_ENTAILMENT]), float(probs[LABEL_NEUTRAL])


class NLIVerifier:
    """Cross-encoder NLI verifier (injectable model for tests)."""

    def __init__(self, model_name: str = DEFAULT_NLI, model: TripletScorer | None = None):
        self.model_name = model_name
        if model is None:
            from sentence_transformers import CrossEncoder

            try:
                model = CrossEncoder(model_name, local_files_only=True)
            except (OSError, ValueError, EnvironmentError):
                model = CrossEncoder(model_name)
        self.model = model

    def _score_pair(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        logits = np.asarray(self.model.predict([(premise, hypothesis)], show_progress_bar=False))
        if logits.ndim == 1 and logits.size == 3:
            return _softmax_probs(logits)
        return _softmax_probs(logits[0])

    def verify(
        self,
        question: str,
        answer: str,
        chunks: list[dict[str, Any]],
        citations: list[int] | None = None,
    ) -> VerificationResult:
        from src.generation.prompts import INSUFFICIENT_EVIDENCE

        if not chunks or answer.strip().lower() == INSUFFICIENT_EVIDENCE:
            return VerificationResult(
                supported=False,
                entailment=0.0,
                contradiction=0.0,
                neutral=1.0,
                premise="",
                hypothesis=answer,
                used_citations=False,
            )

        premise, used_citations = evidence_text(chunks, citations)
        hypothesis = build_hypothesis(question, answer)
        lower = answer.strip().lower()

        start = time.perf_counter()
        rival_entailment = None

        if lower in {"yes", "no"}:
            # Score both sides; keep the answer only if its side is preferred.
            yes_c, yes_e, yes_n = self._score_pair(premise, _as_statement(question))
            no_c, no_e, no_n = self._score_pair(premise, _negated_statement(question))
            if lower == "yes":
                contradiction, entailment, neutral = yes_c, yes_e, yes_n
                rival_entailment = no_e
                supported = yes_e >= no_e and yes_e >= yes_c
            else:
                contradiction, entailment, neutral = no_c, no_e, no_n
                rival_entailment = yes_e
                supported = no_e >= yes_e and no_e >= no_c
        else:
            contradiction, entailment, neutral = self._score_pair(premise, hypothesis)
            # Short HotpotQA spans are often labeled neutral. Reject only when
            # contradiction is the strongest of the three labels.
            supported = contradiction < max(entailment, neutral)

        latency = time.perf_counter() - start
        return VerificationResult(
            supported=supported,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            premise=premise,
            hypothesis=hypothesis,
            used_citations=used_citations,
            latency_s=latency,
            rival_entailment=rival_entailment,
        )
