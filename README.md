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

- Python **3.13** (tested with 3.13.12; pinned versions in `requirements.txt` assume it)
- [Git LFS](https://git-lfs.com/) (required to download the full `.arrow` cache)
- [Ollama](https://ollama.com/) for local answer generation (`brew install ollama`), with the
  `llama3.2:3b` model (~2 GB): `ollama pull llama3.2:3b`
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

Copy `.env.example` to `.env`. All settings are optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOTPOTQA_CACHE_DIR` | `data/raw/huggingface` | Cache directory used by `--source huggingface` |
| `OLLAMA_HOST` | `http://localhost:11434` | Address of the local Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model used for answer generation |

Relative paths resolve from the repository root. No API keys are required.

## Quick start (fresh clone → results)

Run these in order from the repository root after [Setup](#setup). The
sentence chunks (`data/processed/*_chunks.jsonl`) are already in the repo; the
FAISS index is not and must be built once.

```bash
source .venv/bin/activate

# 1. Build the validation FAISS index (~1–2 min; downloads the embedding model, ~90 MB)
python -m src.retrieval.build_index --split validation

# 2. Start the local LLM (separate terminal, keep it running)
ollama serve                   # or: brew services start ollama
ollama pull llama3.2:3b        # first time only (~2 GB)

# 3. Run the tests
python -m pytest

# 4. Try one question (Standard RAG, then Reranked RAG)
python -m src.pipeline.baseline --index 3
python -m src.pipeline.baseline --index 3 --rerank   # first run downloads the reranker, ~90 MB

# 5. Reproduce the evaluation results (~2 min each)
python -m evaluation.run_baseline --n 100            # Standard RAG
python -m evaluation.run_baseline --n 100 --rerank   # Reranked RAG
```

Notes:

- The embedding and reranker models download from Hugging Face on first use,
  so steps 1 and 4 need internet once. After that you can run offline with
  `export HF_HUB_OFFLINE=1`.
- Steps 4–5 fail with "Cannot reach Ollama" if `ollama serve` is not running.
  Check with `curl http://localhost:11434/api/version`.
- Answers use temperature 0, so re-running gives the same results as
  `evaluation/results/` (small differences are possible across Ollama versions).
- Re-chunking is only needed if you change the chunking code:
  `python -m src.ingestion.chunk_hotpotqa --split all`.

### Dependencies

Packages used by the current pipeline are pinned to exact versions in
`requirements.txt`. Packages for later stages keep minimum versions until
they are implemented.

| Group | Packages | Needed for |
| --- | --- | --- |
| Config / data | `python-dotenv`, `datasets` | `.env`, HotpotQA HF cache |
| Processing | `numpy`, `pandas`, `tqdm` | data handling, progress |
| Retrieval | `sentence-transformers`, `faiss-cpu`, `torch`, `scikit-learn` | embeddings + vector search |
| Generation | `httpx` | calls to a local Ollama LLM server |
| Testing | `pytest`, `pytest-cov` | automated tests |
| Planned | `fastapi`, `uvicorn`, `pydantic`, `ragas` | backend service, RAGAS evaluation |

Postgres/`pgvector` is commented out in `requirements.txt` as an optional alternative to FAISS.

## Run the tests

```bash
python -m pytest
python -m pytest --cov=src     # with coverage
```

Chunking, generation, reranking, pipeline, and metrics tests use small fakes,
so they need neither Ollama nor model downloads. Retrieval tests skip
automatically if the FAISS index has not been built yet
(`python -m src.retrieval.build_index --split validation`).

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
│   ├── reranking/
│   │   └── cross_encoder.py    # cross-encoder reranker (top 20 → top 5)
│   ├── generation/
│   │   ├── prompts.py          # grounded prompt with numbered evidence
│   │   └── llm.py              # Ollama client, JSON answer + citations
│   ├── pipeline/
│   │   └── baseline.py         # Standard / Reranked RAG: retrieve → (rerank) → generate
│   ├── verification/           # planned
│   └── adaptive/               # planned
├── evaluation/
│   ├── metrics.py              # EM, F1, precision/recall
│   ├── run_baseline.py         # Standard / Reranked RAG evaluation (--rerank)
│   └── results/                # per-question results + summaries
├── tests/                      # pytest: chunking, retrieval, generation, reranking, pipeline, metrics
├── pytest.ini
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
| `Cannot reach Ollama at http://localhost:11434` | Start `ollama serve` (or `brew services start ollama`) and confirm `ollama list` shows `llama3.2:3b` |
| `FileNotFoundError` for a `.faiss` index | Run `python -m src.retrieval.build_index --split validation` |
| `couldn't connect to 'https://huggingface.co'` | First model download needs internet; unset `HF_HUB_OFFLINE` and retry |

## Status

| Area | Status |
| --- | --- |
| Project scaffold + README | Done |
| Curated HotpotQA JSON in repo | Done |
| Full HotpotQA HF cache via Git LFS | Done |
| Sentence chunking → `data/processed/` | Done |
| Embeddings + FAISS retrieval | Done |
| Answer generation (local Ollama) | Done |
| End-to-end baseline RAG command | Done |
| Baseline evaluation (100 questions) | Done |
| Cross-encoder reranking + Reranked RAG evaluation | Done |
| Verification, confidence scoring | Not started |
| Adaptive retry, three-system comparison | Not started |

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

### Step 3 — grounded answer generation

`src/generation/` sends the retrieved sentences to a local Ollama model as
numbered evidence. The model must answer only from that evidence, return JSON
`{"answer": ..., "citations": [...]}`, and reply `insufficient evidence` when the
answer is not supported. Temperature is 0 so runs are repeatable.

```bash
ollama serve                 # keep running in a separate terminal
ollama pull llama3.2:3b      # first time only
```

Configure the host and model in `.env` (`OLLAMA_HOST`, `OLLAMA_MODEL`).

### Step 4 — end-to-end baseline RAG

`src/pipeline/baseline.py` runs Standard RAG: retrieve the top-k sentences, then
generate one answer. There is no reranking, verification, or retry; this is the
baseline the adaptive system will be compared against.

```bash
# Question from a validation example (searches only that example's context)
python -m src.pipeline.baseline --index 0
python -m src.pipeline.baseline --example-id 5a8b57f25542995d1e6f1371

# Same question, but search the whole validation split
python -m src.pipeline.baseline --index 0 --scope global

# Any free-form question over the whole split
python -m src.pipeline.baseline --question "Who directed the film Ed Wood?"
```

For dataset questions, the output shows the gold answer and marks each retrieved
sentence as cited by the model (`*`) and/or a gold supporting fact (`G`).

Example of the reliability problem this project targets (`--index 3`): retrieval
finds both gold sentences (Ortaköy vs. Laleli), but the baseline still answers
"yes" when the gold answer is "no".

### Step 5 — baseline evaluation

```bash
python -m evaluation.run_baseline --n 100 --top-k 5
```

Runs Standard RAG on the first N validation questions (distractor setting:
each question searches only its own context) and writes per-question results
and a summary to `evaluation/results/`. Answer metrics use the official
HotpotQA normalization.

**Baseline results** — 100 validation questions, `llama3.2:3b`, MiniLM
embeddings, top-k = 5 (`evaluation/results/baseline_n100_k5_summary.json`):

| Metric | Value |
| --- | ---: |
| Answer exact match | 40.0% |
| Answer F1 | 50.1% |
| Answered "insufficient evidence" | 12.0% |
| Retrieval recall@5 (gold sentences found) | 59.5% |
| All gold sentences retrieved | 33.0% |
| Citation precision / recall | 56.2% / 36.0% |
| Average latency | 1.02 s (retrieval 0.02 s, generation 0.99 s) |
| Average tokens | 365 prompt + 17 completion |

| Question type | n | EM | F1 | Retrieval recall |
| --- | ---: | ---: | ---: | ---: |
| Bridge | 79 | 35.4% | 44.5% | 55.8% |
| Comparison | 21 | 57.1% | 71.2% | 73.4% |

What this shows:

- When all gold sentences were retrieved (33 questions), exact match was 72.7%.
  When some were missing (67 questions), it fell to 23.9%.
- With incomplete evidence, the model abstained only 12 times and gave a
  confident wrong answer 39 times. Overall, 48 of 100 answers were confidently
  wrong.
- Even with complete evidence, 9 answers were still wrong (generation errors,
  such as the Laleli Mosque example).

These are the two failure modes the adaptive system targets: detecting weak
evidence and retrieving more, and verifying answers against the evidence.

### Step 6 — Reranked RAG (cross-encoder)

`src/reranking/cross_encoder.py` adds a second retrieval stage. FAISS fetches
20 candidate sentences, then `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each
(question, "title: sentence") pair jointly and keeps the top 5 for generation.
The model downloads on first use (~90 MB).

```bash
python -m src.pipeline.baseline --index 3 --rerank
python -m evaluation.run_baseline --n 100 --rerank --candidate-k 20
```

**Standard RAG vs. Reranked RAG** — same 100 validation questions, same
`llama3.2:3b` generator, top-k = 5
(`evaluation/results/reranked_n100_k5_summary.json`):

| Metric | Standard RAG | Reranked RAG | Change |
| --- | ---: | ---: | ---: |
| Answer exact match | 40.0% | 44.0% | +4.0 |
| Answer F1 | 50.1% | 55.7% | +5.6 |
| Retrieval recall@5 | 59.5% | 71.7% | +12.2 |
| All gold sentences retrieved | 33.0% | 45.0% | +12.0 |
| Citation precision | 56.2% | 65.5% | +9.3 |
| Citation recall | 36.0% | 43.2% | +7.2 |
| Answered "insufficient evidence" | 12.0% | 9.0% | −3.0 |
| Average latency | 1.02 s | 1.12 s | +0.10 s |

| Question type | Standard EM / F1 | Reranked EM / F1 | Recall@5 (std → reranked) |
| --- | ---: | ---: | ---: |
| Bridge (n=79) | 35.4% / 44.5% | 39.2% / 49.5% | 55.8% → 66.0% |
| Comparison (n=21) | 57.1% / 71.2% | 61.9% / 79.2% | 73.4% → 93.2% |

What this shows:

- Reranking mainly fixes retrieval: 12 more questions get all their gold
  sentences, for only ~0.04 s extra per question.
- Answer accuracy improves less than retrieval. Reranking turned 11 wrong
  answers into correct ones but 7 correct answers into wrong ones.
- With complete evidence, the generator is still wrong on 16 of 45 questions
  (EM 64.4%). The Laleli Mosque example now ranks both gold sentences first and
  second, but the model still answers "yes".
- 47 of 100 answers are still confidently wrong (48 for the baseline). Better
  ranking alone does not make the system know when it is wrong.

This motivates the next stages: verifying the answer against the evidence and
scoring confidence, so weak or unsupported answers can be retried or abstained.

Next milestone: evidence verification and confidence scoring.
