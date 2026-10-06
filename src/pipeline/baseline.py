"""Standard (baseline) RAG: retrieve top-k sentences, then generate one answer.

No reranking, verification, or retries. This is the baseline the adaptive
system will be compared against.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from src.generation.llm import GeneratedAnswer, OllamaGenerator
from src.ingestion.load_hotpotqa import load_hotpotqa_local
from src.retrieval.search import FaissRetriever, SearchHit


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[int]
    evidence: list[SearchHit]
    retrieval_s: float
    generation_s: float
    generated: GeneratedAnswer

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.generation_s


class BaselineRAG:
    def __init__(self, retriever: FaissRetriever, generator: OllamaGenerator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str, example_id: str | None = None) -> RAGResult:
        start = time.perf_counter()
        hits = self.retriever.search(question, top_k=self.top_k, example_id=example_id)
        retrieval_s = time.perf_counter() - start

        generated = self.generator.generate(question, [h.chunk for h in hits])
        return RAGResult(
            question=question,
            answer=generated.answer,
            citations=generated.citations,
            evidence=hits,
            retrieval_s=retrieval_s,
            generation_s=generated.latency_s,
            generated=generated,
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
    if example:
        print(f"Gold:     {example['answer']}")
    print(f"Time:     retrieval {result.retrieval_s:.2f}s + generation {result.generation_s:.2f}s")

    print(f"\nRetrieved evidence (top {len(result.evidence)}; * = cited by model, G = gold supporting fact):")
    for i, hit in enumerate(result.evidence, start=1):
        c = hit.chunk
        cited = "*" if i in result.citations else " "
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

    rag = BaselineRAG(FaissRetriever(split=args.split), OllamaGenerator(), top_k=args.top_k)
    print_result(rag.answer(question, example_id=example_filter), example)


if __name__ == "__main__":
    main()
