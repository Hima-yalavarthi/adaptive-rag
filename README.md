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

All project stages are listed in `requirements.txt`:

| Group | Packages | Needed for |
| --- | --- | --- |
| Config / data | `python-dotenv`, `datasets` | `.env`, HotpotQA HF cache |
| Processing | `numpy`, `pandas`, `tqdm`, `tiktoken` | data handling, progress, token/cost tracking |
| Retrieval | `sentence-transformers`, `faiss-cpu`, `torch` | embeddings + vector search |
| Generation | `openai`, `httpx`, `tenacity` | LLM API calls with retries |
| API | `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings` | backend service |
| Evaluation | `ragas`, `scikit-learn` | faithfulness / quality metrics |
| Testing | `pytest`, `pytest-cov` | automated tests |

Postgres/`pgvector` is commented out in `requirements.txt` as an optional alternative to FAISS.

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
│   │   ├── load_hotpotqa.py    # local JSON + huggingface cache loaders
│   │   └── chunk_hotpotqa.py   # sentence-level context chunking
│   ├── retrieval/
│   │   ├── build_index.py      # embed chunks + FAISS index
│   │   └── search.py           # semantic search CLI
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
    └── processed/
        ├── hotpotqa_*_chunks.jsonl
        └── indexes/            # FAISS indexes (rebuild locally)
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
| Sentence chunking → `data/processed/` | Done |
| Embeddings + FAISS retrieval | Done |
| Reranking, generation, verification | Not started |
| Adaptive retry, evaluation | Not started |

### Step 1 — chunk HotpotQA contexts

```bash
python -m src.ingestion.chunk_hotpotqa --split validation
python -m src.ingestion.chunk_hotpotqa --split all
```

Writes sentence-level chunks to `data/processed/hotpotqa_<split>_chunks.jsonl`
(`chunk_id`, `example_id`, `title`, `sent_id`, `text`). These map to HotpotQA
supporting-fact annotations for later evaluation.

### Step 2 — embed chunks and search with FAISS

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized cosine via
inner product).

```bash
# Build indexes (downloads the embedding model on first run)
python -m src.retrieval.build_index --split validation
python -m src.retrieval.build_index --split train

# Semantic search
python -m src.retrieval.search --split validation \
  --query "Were Scott Derrickson and Ed Wood of the same nationality?" --top-k 5

# Restrict to one HotpotQA example's context (distractor-style)
python -m src.retrieval.search --split validation \
  --query "Were Scott Derrickson and Ed Wood of the same nationality?" \
  --example-id 5a8b57f25542995d1e6f1371 --top-k 5
```

Index files under `data/processed/indexes/` are large and Git-ignored; rebuild
locally after cloning. Config JSON files record model name and vector counts.

Next milestone: LLM answer generation for a baseline RAG pipeline.
