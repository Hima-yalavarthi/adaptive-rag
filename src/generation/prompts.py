"""Build grounded QA prompts from retrieved evidence sentences."""

from __future__ import annotations

from typing import Any

INSUFFICIENT_EVIDENCE = "insufficient evidence"

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered evidence sentences provided.

Rules:
- Give the shortest possible answer: a name, number, date, short phrase, or "yes"/"no".
- Cite the evidence numbers you used.
- Do not use outside knowledge.
- If the evidence does not contain the answer, set the answer to "{INSUFFICIENT_EVIDENCE}" and cite nothing.

Respond with JSON only: {{"answer": "<short answer>", "citations": [<evidence numbers>]}}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "citations"],
}


def format_evidence(chunks: list[dict[str, Any]]) -> str:
    """Number evidence sentences starting at 1, prefixed with their article title."""
    return "\n".join(
        f"[{i}] ({chunk['title']}) {chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    return f"Evidence:\n{format_evidence(chunks)}\n\nQuestion: {question}"
