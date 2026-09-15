# Adaptive RAG for Reliable Question Answering

Capstone project (CAI 6826, University of Florida) — a RAG system that checks
whether generated answers are supported by retrieved evidence, scores confidence,
and retries retrieval when support is weak. If evidence remains insufficient, the
system returns an insufficient-evidence response instead of an unsupported answer.

**Author:** Hima Yalavarthi · **Type:** Individual project  
**Repository:** https://github.com/Hima-yalavarthi/adaptive-rag

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

## Dataset overview

This project uses [HotpotQA](https://hotpotqa.github.io/) (`hotpotqa/hotpot_qa`,
`distractor` configuration). Each example includes a question, answer, context
passages, and supporting-fact annotations.

HotpotQA is distributed under the
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
license.

Data is available in the repo in **two forms**:

| Source | Path | Examples | Approx. size | Tracked how | Best for |
| --- | --- | ---: | ---: | --- | --- |
| Curated JSON | `data/raw/hotpotqa_train.json` | 2,000 | ~13.5 MB | Regular Git | Fast development / demos |
| Curated JSON | `data/raw/hotpotqa_validation.json` | 500 | ~3.4 MB | Regular Git | Quick validation checks |
| Full HF cache | `data/raw/huggingface/**/*.arrow` | full train (~90k) + validation (~7.4k) | ~599 MB | **Git LFS** | Full-data experiments |

### Why both?

- GitHub rejects normal Git files over **100 MB**. One HotpotQA train shard is ~479 MB.
- **Git LFS** stores those large `.arrow` files so the full dataset can live in this repo.
- The curated JSON files stay small so you can iterate without always fetching LFS objects.

### Full Hugging Face cache contents

```text
data/raw/huggingface/hotpotqa___hotpot_qa/distractor/0.0.0/<hash>/
├── dataset_info.json
├── hotpot_qa-train-00000-of-00002.arrow   # Git LFS
├── hotpot_qa-train-00001-of-00002.arrow   # Git LFS
└── hotpot_qa-validation.arrow             # Git LFS
```

LFS tracking is defined in `.gitattributes`:

```text
data/raw/huggingface/**/*.arrow filter=lfs diff=lfs merge=lfs -text
```

## Prerequisites

- Python **3.11+**
- [Git LFS](https://git-lfs.com/) (required to download the full `.arrow` cache)
- ~1 GB free disk for a full clone with LFS objects

Install Git LFS once on your machine:

```bash
# macOS
brew install git-lfs
git lfs install

# Linux (example)
# sudo apt install git-lfs && git lfs install
```

## Clone

```bash
git clone https://github.com/Hima-yalavarthi/adaptive-rag.git
cd adaptive-rag
```

If you already cloned **without** LFS installed, fetch the large files afterward:

```bash
git lfs install
git lfs pull
```

To clone **without** downloading ~599 MB of LFS data (JSON-only workflow):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Hima-yalavarthi/adaptive-rag.git
cd adaptive-rag
# later, when you need the full dataset:
git lfs pull
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env               # Windows: Copy-Item .env.example .env
```

### Environment variables

Copy `.env.example` to `.env`. Optional setting:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOTPOTQA_CACHE_DIR` | `data/raw/huggingface` | Cache directory used by `--source huggingface` |

Relative paths resolve from the repository root. No API key is required for HotpotQA.

### Dependencies

| Package | Needed for |
| --- | --- |
| `python-dotenv` | Loading `.env` |
| `datasets` | Full Hugging Face cache loader (`--source huggingface`) |
| `sentence-transformers`, `faiss-cpu`, `fastapi`, `uvicorn` | Planned later stages |

## Load / verify the dataset

From the repository root, with the virtualenv activated:

```bash
# Curated subset (default) — uses data/raw/*.json
python -m src.ingestion.load_hotpotqa
python -m src.ingestion.load_hotpotqa --source local
python -m src.ingestion.load_hotpotqa --source local --split train
python -m src.ingestion.load_hotpotqa --source local --split validation

# Full HotpotQA — uses the Git LFS Hugging Face cache
python -m src.ingestion.load_hotpotqa --source huggingface
python -m src.ingestion.load_hotpotqa --source huggingface --split train
python -m src.ingestion.load_hotpotqa --source huggingface --split validation
```

Expected curated totals: **2,000** train + **500** validation = **2,500** examples.  
Expected full distractor totals: **90,447** train + **7,405** validation = **97,852** examples.

## Project structure

```text
adaptive-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitattributes              # Git LFS rules for *.arrow
├── .gitignore
├── src/
│   ├── ingestion/
│   │   └── load_hotpotqa.py    # local JSON + huggingface cache loaders
│   ├── retrieval/              # planned
│   ├── reranking/              # planned
│   ├── generation/             # planned
│   ├── verification/           # planned
│   └── adaptive/               # planned
├── evaluation/                 # planned
├── tests/                      # planned
├── notebooks/
└── data/
    ├── raw/
    │   ├── hotpotqa_train.json
    │   ├── hotpotqa_validation.json
    │   └── huggingface/        # full HF cache (*.arrow via Git LFS)
    └── processed/              # future transformed data
```

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `.arrow` files are tiny pointer files after clone | Install Git LFS, run `git lfs install`, then `git lfs pull` |
| `datasets` / import errors on `--source huggingface` | `pip install -r requirements.txt` inside the active `.venv` |
| Missing curated JSON | Confirm you are in the repo root and files exist under `data/raw/` |
| LFS quota / bandwidth errors on GitHub | Use curated JSON for daily work; pull LFS only when needed |
| Want to skip large download on clone | Use `GIT_LFS_SKIP_SMUDGE=1 git clone ...` then `git lfs pull` later |

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
