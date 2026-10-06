"""Generate grounded answers with a local Ollama model."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from src.generation.prompts import (
    ANSWER_SCHEMA,
    INSUFFICIENT_EVIDENCE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.ingestion.load_hotpotqa import PROJECT_ROOT

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[int]
    cited_chunks: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""

    @property
    def is_insufficient(self) -> bool:
        return self.answer.strip().lower() == INSUFFICIENT_EVIDENCE


def parse_answer(raw: str, num_chunks: int) -> tuple[str, list[int]]:
    """Parse the model's JSON reply, dropping citations outside 1..num_chunks."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or INSUFFICIENT_EVIDENCE, []

    answer = str(data.get("answer", "")).strip() or INSUFFICIENT_EVIDENCE
    citations: list[int] = []
    for c in data.get("citations") or []:
        try:
            n = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= num_chunks and n not in citations:
            citations.append(n)
    return answer, citations


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout_s: float = 120.0,
    ):
        load_dotenv(PROJECT_ROOT / ".env")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> GeneratedAnswer:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, chunks)},
            ],
            "format": ANSWER_SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        }

        start = time.perf_counter()
        try:
            response = self.client.post(f"{self.host}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from exc
        response.raise_for_status()
        latency = time.perf_counter() - start

        body = response.json()
        raw = body.get("message", {}).get("content", "")
        answer, citations = parse_answer(raw, len(chunks))
        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            cited_chunks=[chunks[n - 1] for n in citations],
            latency_s=latency,
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            raw=raw,
        )
