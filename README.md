# Adaptive RAG for Reliable Question Answering

Capstone project (CAI 6826, University of Florida) — a RAG system that checks
whether generated answers are supported by retrieved evidence, scores confidence,
and retries retrieval when support is weak. If evidence remains insufficient, the
system returns an insufficient-evidence response instead of an unsupported answer.

**Author:** Hima Yalavarthi · **Type:** Individual project

## Architecture (planned)

```text
User Question
  → Query Analysis
  → Semantic Retrieval
  → Reranking
  → LLM Answer Generation
  → Evidence Verification
  → Confidence Scoring
  → Adaptive Retry  or  Final Answer + Evidence
```

## Comparison plan

| System | Behavior |
| --- | --- |
| Standard RAG | Retrieve + generate (baseline) |
| Reranked RAG | Retrieve + rerank + generate |
| Adaptive RAG | Above + verification, confidence, bounded retries |

## Dataset

[HotpotQA](https://hotpotqa.github.io/) (`hotpotqa/hotpot_qa`, `distractor` config)
via Hugging Face. Multi-hop Wikipedia QA with supporting-fact annotations.
Development will use a manageable subset; full splits are available for access checks.

## Project structure

```text
adaptive-rag/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/          # HotpotQA loader (implemented)
│   ├── retrieval/          # planned
│   ├── reranking/          # planned
│   ├── generation/         # planned
│   ├── verification/       # planned
│   └── adaptive/           # planned
├── evaluation/             # planned
├── tests/                  # planned
├── notebooks/
└── data/
    ├── raw/                # local downloads (gitignored)
    └── processed/
```

## Setup

Python 3.11+ recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Optional: set `HOTPOTQA_CACHE_DIR` in `.env` (default: `data/raw/huggingface`).

## Confirm dataset access

```bash
python -m src.ingestion.load_hotpotqa
python -m src.ingestion.load_hotpotqa --split validation
```

First run needs internet and downloads into the local cache. No API key required.

## Status

| Area | Status |
| --- | --- |
| Project scaffold + README | Done |
| HotpotQA load / access check | Done |
| Chunking, embeddings, retrieval | Not started |
| Reranking, generation, verification | Not started |
| Adaptive retry, evaluation | Not started |

Next milestone: document preprocessing, embeddings, and a baseline RAG pipeline.
