"""Standard / Reranked / Verified RAG: retrieve, optionally rerank, generate, optionally verify.

- Standard RAG: take the FAISS top-k sentences.
- Reranked RAG (--rerank): take the FAISS top candidate-k, rerank with a
  cross-encoder, keep the top-k.
- Verified RAG (--verify): after generation, NLI-check the answer against
  evidence and abstain when confidence is below the threshold.

Adaptive retry (query rewrite / more candidates) is the next stage.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from src.generation.llm import GeneratedAnswer, OllamaGenerator
from src.generation.prompts import INSUFFICIENT_EVIDENCE
from src.ingestion.load_hotpotqa import load_hotpotqa_local
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.search import FaissRetriever, SearchHit
from src.verification.confidence import DEFAULT_THRESHOLD, ConfidenceResult, score_confidence
from src.verification.nli import NLIVerifier, VerificationResult


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[int]
    evidence: list[SearchHit]
    retrieval_s: float
    generation_s: float
    generated: GeneratedAnswer
    rerank_s: float = 0.0
    verify_s: float = 0.0
    verification: VerificationResult | None = None
    confidence: ConfidenceResult | None = None
    abstained_by_verifier: bool = False
    raw_answer: str = ""

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.rerank_s + self.generation_s + self.verify_s


class BaselineRAG:
    def __init__(
        self,
        retriever: FaissRetriever,
        generator: OllamaGenerator,
        top_k: int = 5,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 20,
        verifier: NLIVerifier | None = None,
        confidence_threshold: float = DEFAULT_THRESHOLD,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.reranker = reranker
        self.candidate_k = max(candidate_k, top_k)
        self.verifier = verifier
        self.confidence_threshold = confidence_threshold

    def answer(self, question: str, example_id: str | None = None) -> RAGResult:
        fetch_k = self.candidate_k if self.reranker else self.top_k
        start = time.perf_counter()
        hits = self.retriever.search(question, top_k=fetch_k, example_id=example_id)
        retrieval_s = time.perf_counter() - start

        rerank_s = 0.0
        if self.reranker:
            start = time.perf_counter()
            hits = self.reranker.rerank(question, hits, top_k=self.top_k)
            rerank_s = time.perf_counter() - start

        generated = self.generator.generate(question, [h.chunk for h in hits])
        answer = generated.answer
        citations = list(generated.citations)
        raw_answer = generated.answer
        verify_s = 0.0
        verification = None
        confidence = None
        abstained = False

        if self.verifier and not generated.is_insufficient:
            verification = self.verifier.verify(
                question, generated.answer, [h.chunk for h in hits], generated.citations
            )
            verify_s = verification.latency_s
            confidence = score_confidence(
                verification,
                hits,
                generated.citations,
                threshold=self.confidence_threshold,
            )
            if confidence.reject:
                answer = INSUFFICIENT_EVIDENCE
                citations = []
                abstained = True

        return RAGResult(
            question=question,
            answer=answer,
            citations=citations,
            evidence=hits,
            retrieval_s=retrieval_s,
            rerank_s=rerank_s,
            generation_s=generated.latency_s,
            verify_s=verify_s,
            generated=generated,
            verification=verification,
            confidence=confidence,
            abstained_by_verifier=abstained,
            raw_answer=raw_answer,
        )


def gold_facts(example: dict[str, Any]) -> set[tuple[str, int]]:
    facts = example["supporting_facts"]
    return set(zip(facts["title"], facts["sent_id"]))


def find_example(split: str, example_id: str | None, index: int | None) -> dict[str, Any]:
    examples = load_hotpotqa_local(split)
    if example_id is not None:
        for ex in examples:
            if ex["id"] == example_id:
                return ex
        raise ValueError(f"example_id {example_id} not found in {split}")
    return examples[index]


def print_result(result: RAGResult, example: dict[str, Any] | None) -> None:
    gold = gold_facts(example) if example else set()

    print(f"Question: {result.question}")
    print(f"Answer:   {result.answer}")
    if result.abstained_by_verifier:
        print(f"Raw LLM:  {result.raw_answer}  (abstained by verifier)")
    if example:
        print(f"Gold:     {example['answer']}")
    parts = [f"retrieval {result.retrieval_s:.2f}s"]
    if result.rerank_s:
        parts.append(f"rerank {result.rerank_s:.2f}s")
    parts.append(f"generation {result.generation_s:.2f}s")
    if result.verify_s:
        parts.append(f"verify {result.verify_s:.2f}s")
    print(f"Time:     {' + '.join(parts)}")

    if result.confidence is not None and result.verification is not None:
        c, v = result.confidence, result.verification
        decision = "ACCEPT" if c.accept else "ABSTAIN"
        print(
            f"Verify:   {decision}  confidence={c.score:.3f} "
            f"(threshold={c.threshold:.2f}; entail={v.entailment:.3f} "
            f"contradict={v.contradiction:.3f} neutral={v.neutral:.3f})"
        )
        print(f"Hypothesis: {v.hypothesis}")

    print(f"\nRetrieved evidence (top {len(result.evidence)}; * = cited by model, G = gold supporting fact):")
    for i, hit in enumerate(result.evidence, start=1):
        c = hit.chunk
        cited = "*" if i in (result.generated.citations if result.abstained_by_verifier else result.citations) else " "
        is_gold = "G" if (c["title"], c["sent_id"]) in gold else " "
        print(f" {cited}{is_gold} [{i}] {hit.score:.3f} ({c['title']}) {c['text']}")

    if example:
        found = {(h.chunk["title"], h.chunk["sent_id"]) for h in result.evidence} & gold
        print(f"\nGold supporting facts retrieved: {len(found)}/{len(gold)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="Ask any question over the whole split.")
    source.add_argument("--example-id", help="Use the question from this HotpotQA example.")
    source.add_argument("--index", type=int, help="Use the question from the N-th example in the split.")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Rerank candidates with a cross-encoder.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidates retrieved before reranking.")
    parser.add_argument("--verify", action="store_true", help="Verify the answer with NLI and abstain if weak.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Abstain when confidence is below this (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--scope",
        choices=("example", "global"),
        default="example",
        help="For dataset questions: search only that example's context (example) or the whole split (global).",
    )
    args = parser.parse_args()

    example = None
    question = args.question
    example_filter = None
    if args.question is None:
        example = find_example(args.split, args.example_id, args.index)
        question = example["question"]
        if args.scope == "example":
            example_filter = example["id"]

    rag = BaselineRAG(
        FaissRetriever(split=args.split),
        OllamaGenerator(),
        top_k=args.top_k,
        reranker=CrossEncoderReranker() if args.rerank else None,
        candidate_k=args.candidate_k,
        verifier=NLIVerifier() if args.verify else None,
        confidence_threshold=args.confidence_threshold,
    )
    print_result(rag.answer(question, example_id=example_filter), example)


if __name__ == "__main__":
    main()
