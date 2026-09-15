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

Curated [HotpotQA](https://hotpotqa.github.io/) distractor subsets are stored in
the repo under `data/raw/`:

| File | Examples |
| --- | ---: |
| `data/raw/hotpotqa_train.json` | 2,000 |
| `data/raw/hotpotqa_validation.json` | 500 |

Each record includes question, answer, context passages, and supporting facts.
The full Hugging Face download is too large for GitHub; these subsets match the
proposal’s planned development scale.

## Project structure

```text
adaptive-rag/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/          # HotpotQA JSON loader (implemented)
│   ├── retrieval/          # planned
│   ├── reranking/          # planned
│   ├── generation/         # planned
│   ├── verification/       # planned
│   └── adaptive/           # planned
├── evaluation/             # planned
├── tests/                  # planned
├── notebooks/
└── data/
    ├── raw/                # curated HotpotQA JSON (in repo)
    └── processed/          # future transformed data
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

## Confirm dataset access

```bash
python -m src.ingestion.load_hotpotqa
python -m src.ingestion.load_hotpotqa --split validation
```

No download or API key is required — data is read from `data/raw/*.json`.

## Status

| Area | Status |
| --- | --- |
| Project scaffold + README | Done |
| Curated HotpotQA data in repo | Done |
| Chunking, embeddings, retrieval | Not started |
| Reranking, generation, verification | Not started |
| Adaptive retry, evaluation | Not started |

Next milestone: document preprocessing, embeddings, and a baseline RAG pipeline.
