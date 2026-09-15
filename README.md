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

[HotpotQA](https://hotpotqa.github.io/) distractor data is stored two ways:

| Source | Location | Size | Use when |
| --- | --- | --- | --- |
| Curated JSON | `data/raw/hotpotqa_*.json` | 2,000 train + 500 validation | Fast local iteration |
| Full HF cache | `data/raw/huggingface/` (Git LFS) | full train + validation | Full-data experiments |

Arrow files in the Hugging Face cache are tracked with **Git LFS**. Clone with LFS
installed (`brew install git-lfs && git lfs install`) so the full cache downloads.

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
    ├── raw/                # curated JSON + full HF cache (LFS)
    └── processed/          # future transformed data
```

## Setup

Requires [Git LFS](https://git-lfs.com/) for the full Hugging Face cache.

```bash
brew install git-lfs   # if needed
git lfs install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Confirm dataset access

```bash
# Curated subset (default, no LFS fetch required for day-to-day work)
python -m src.ingestion.load_hotpotqa
python -m src.ingestion.load_hotpotqa --source local --split validation

# Full HotpotQA from the Git LFS Hugging Face cache
python -m src.ingestion.load_hotpotqa --source huggingface
python -m src.ingestion.load_hotpotqa --source huggingface --split train
```

## Status

| Area | Status |
| --- | --- |
| Project scaffold + README | Done |
| Curated HotpotQA JSON in repo | Done |
| Full HotpotQA HF cache via Git LFS | Done |
| Chunking, embeddings, retrieval | Not started |
| Reranking, generation, verification | Not started |
| Adaptive retry, evaluation | Not started |

Next milestone: document preprocessing, embeddings, and a baseline RAG pipeline.
