"""Evaluate Standard / Reranked / Verified RAG on HotpotQA validation questions.

Uses the distractor setting: each question retrieves only from its own
example's context. Writes per-question results and a summary to
evaluation/results/.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from evaluation.metrics import exact_match, f1_score, set_precision_recall
from src.generation.llm import OllamaGenerator
from src.generation.prompts import INSUFFICIENT_EVIDENCE
from src.ingestion.load_hotpotqa import PROJECT_ROOT, load_hotpotqa_local
from src.pipeline.baseline import BaselineRAG, gold_facts
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.search import FaissRetriever
from src.verification.confidence import DEFAULT_THRESHOLD
from src.verification.nli import NLIVerifier

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def evaluate_example(rag: BaselineRAG, example: dict) -> dict:
    result = rag.answer(example["question"], example_id=example["id"])
    gold = gold_facts(example)

    retrieved = {(h.chunk["title"], h.chunk["sent_id"]) for h in result.evidence}
    cited_source = result.generated.cited_chunks
    cited = {(c["title"], c["sent_id"]) for c in cited_source}
    _, retrieval_recall = set_precision_recall(retrieved, gold)
    citation_precision, citation_recall = set_precision_recall(cited, gold)

    raw = result.raw_answer or result.answer
    raw_em = exact_match(raw, example["answer"])
    final_insufficient = result.answer.strip().lower() == INSUFFICIENT_EVIDENCE

    row = {
        "id": example["id"],
        "type": example["type"],
        "level": example["level"],
        "question": example["question"],
        "gold_answer": example["answer"],
        "prediction": result.answer,
        "raw_prediction": raw,
        "em": exact_match(result.answer, example["answer"]),
        "f1": f1_score(result.answer, example["answer"]),
        "raw_em": raw_em,
        "insufficient": final_insufficient,
        "abstained_by_verifier": result.abstained_by_verifier,
        "retrieval_recall": retrieval_recall,
        "all_gold_retrieved": float(gold <= retrieved),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "retrieval_s": result.retrieval_s,
        "rerank_s": result.rerank_s,
        "generation_s": result.generation_s,
        "verify_s": result.verify_s,
        "prompt_tokens": result.generated.prompt_tokens,
        "completion_tokens": result.generated.completion_tokens,
        "confidence": result.confidence.score if result.confidence else None,
        "entailment": result.verification.entailment if result.verification else None,
        "contradiction": result.verification.contradiction if result.verification else None,
        # Reliability: confidently wrong = wrong final answer that is not an abstention.
        "confidently_wrong": float(not final_insufficient and not exact_match(result.answer, example["answer"])),
        # Verifier caught a wrong raw answer by abstaining.
        "caught_wrong": float(result.abstained_by_verifier and not raw_em),
        # Verifier abstained on a raw answer that was actually correct.
        "false_abstain": float(result.abstained_by_verifier and raw_em),
    }
    return row


METRICS = (
    "em",
    "f1",
    "raw_em",
    "insufficient",
    "abstained_by_verifier",
    "confidently_wrong",
    "caught_wrong",
    "false_abstain",
    "retrieval_recall",
    "all_gold_retrieved",
    "citation_precision",
    "citation_recall",
    "retrieval_s",
    "rerank_s",
    "generation_s",
    "verify_s",
    "prompt_tokens",
    "completion_tokens",
)


def summarize(rows: list[dict]) -> dict:
    def mean(key: str, subset: list[dict]) -> float:
        vals = [float(r[key]) for r in subset if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else 0.0

    summary = {"n": len(rows), "overall": {k: mean(k, rows) for k in METRICS}}
    # Optional continuous signals.
    for key in ("confidence", "entailment", "contradiction"):
        if any(r.get(key) is not None for r in rows):
            summary["overall"][key] = mean(key, rows)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    summary["by_type"] = {
        t: {"n": len(subset), **{k: mean(k, subset) for k in ("em", "f1", "retrieval_recall", "confidently_wrong")}}
        for t, subset in sorted(by_type.items())
    }
    return summary


def print_summary(summary: dict, config: dict) -> None:
    o = summary["overall"]
    if config.get("verifier"):
        title = "Verified RAG"
    elif config.get("reranker"):
        title = "Reranked RAG"
    else:
        title = "Standard RAG"
    print(f"\n{title} on {summary['n']} HotpotQA validation questions")
    print(f"model={config['model']}  top_k={config['top_k']}  embedding={config['embedding_model']}")
    if config.get("reranker"):
        print(f"reranker={config['reranker']}  candidate_k={config['candidate_k']}")
    if config.get("verifier"):
        print(
            f"verifier={config['verifier']}  confidence_threshold={config['confidence_threshold']}"
        )
    print()
    print(f"  Answer exact match      {o['em']:.1%}")
    print(f"  Answer F1               {o['f1']:.1%}")
    print(f"  'Insufficient evidence' {o['insufficient']:.1%}")
    if config.get("verifier"):
        print(f"  Raw EM (before verify)  {o['raw_em']:.1%}")
        print(f"  Abstained by verifier   {o['abstained_by_verifier']:.1%}")
        print(f"  Confidently wrong       {o['confidently_wrong']:.1%}")
        print(f"  Caught wrong answers    {o['caught_wrong']:.1%}")
        print(f"  False abstentions       {o['false_abstain']:.1%}")
        if "confidence" in o:
            print(f"  Avg confidence          {o['confidence']:.3f}")
    print(f"  Retrieval recall@{config['top_k']}      {o['retrieval_recall']:.1%}  (gold supporting sentences found)")
    print(f"  All gold retrieved      {o['all_gold_retrieved']:.1%}")
    print(f"  Citation precision      {o['citation_precision']:.1%}  (cited sentences that are gold)")
    print(f"  Citation recall         {o['citation_recall']:.1%}  (gold sentences the model cited)")
    total = o["retrieval_s"] + o["rerank_s"] + o["generation_s"] + o["verify_s"]
    print(
        f"  Avg latency             {total:.2f}s (retrieval {o['retrieval_s']:.2f}s, "
        f"rerank {o['rerank_s']:.2f}s, generation {o['generation_s']:.2f}s, verify {o['verify_s']:.2f}s)"
    )
    print(f"  Avg tokens              {o['prompt_tokens']:.0f} prompt + {o['completion_tokens']:.0f} completion")
    print("\n  By question type:")
    for t, s in summary["by_type"].items():
        print(f"    {t:<11} n={s['n']:<4} EM {s['em']:.1%}  F1 {s['f1']:.1%}  recall {s['retrieval_recall']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of validation questions (default: 100).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Rerank candidates with a cross-encoder.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidates retrieved before reranking.")
    parser.add_argument("--verify", action="store_true", help="Verify answers with NLI and abstain if weak.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Abstain when confidence is below this (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument("--name", default=None, help="Prefix for output files.")
    args = parser.parse_args()
    if args.name is None:
        if args.verify:
            args.name = "verified"
        elif args.rerank:
            args.name = "reranked"
        else:
            args.name = "baseline"

    examples = load_hotpotqa_local("validation")[: args.n]
    retriever = FaissRetriever(split="validation")
    generator = OllamaGenerator()
    reranker = CrossEncoderReranker() if args.rerank else None
    verifier = NLIVerifier() if args.verify else None
    rag = BaselineRAG(
        retriever,
        generator,
        top_k=args.top_k,
        reranker=reranker,
        candidate_k=args.candidate_k,
        verifier=verifier,
        confidence_threshold=args.confidence_threshold,
    )

    rows = [evaluate_example(rag, ex) for ex in tqdm(examples, desc="Evaluating")]
    config = {
        "system": (
            "verified_rag" if verifier else "reranked_rag" if reranker else "standard_rag"
        ),
        "split": "validation",
        "scope": "example",
        "n": len(rows),
        "top_k": args.top_k,
        "model": generator.model,
        "embedding_model": retriever.model_name,
        "reranker": reranker.model_name if reranker else None,
        "candidate_k": args.candidate_k if reranker else None,
        "verifier": verifier.model_name if verifier else None,
        "confidence_threshold": args.confidence_threshold if verifier else None,
    }
    summary = {"config": config, **summarize(rows)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_question = RESULTS_DIR / f"{args.name}_n{len(rows)}_k{args.top_k}.jsonl"
    summary_path = RESULTS_DIR / f"{args.name}_n{len(rows)}_k{args.top_k}_summary.json"
    with per_question.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary, config)
    print(f"\nSaved: {per_question.relative_to(PROJECT_ROOT)}")
    print(f"       {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
