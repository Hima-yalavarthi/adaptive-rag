"""Build sentence embeddings and a FAISS index from HotpotQA chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ingestion.load_hotpotqa import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROCESSED_DIR / "indexes"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_path(split: str) -> Path:
    return PROCESSED_DIR / f"hotpotqa_{split}_chunks.jsonl"


def index_paths(split: str) -> tuple[Path, Path, Path]:
    """Return (faiss index, metadata jsonl, config json) paths."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return (
        INDEX_DIR / f"hotpotqa_{split}.faiss",
        INDEX_DIR / f"hotpotqa_{split}_meta.jsonl",
        INDEX_DIR / f"hotpotqa_{split}_config.json",
    )


def load_chunks(split: str) -> list[dict]:
    path = chunk_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.ingestion.chunk_hotpotqa --split {split}"
        )
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise RuntimeError(f"No chunks found in {path}")
    return chunks


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(vectors)


def build_index(
    split: str,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> tuple[Path, Path, Path]:
    chunks = load_chunks(split)
    texts = [c["text"] for c in chunks]

    print(f"Split: {split}")
    print(f"Chunks: {len(chunks):,}")
    print(f"Model: {model_name}")

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except (OSError, ValueError, EnvironmentError):
        model = SentenceTransformer(model_name)
    embeddings = embed_texts(model, texts, batch_size=batch_size)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path, meta_path, config_path = index_paths(split)
    faiss.write_index(index, str(faiss_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    config = {
        "split": split,
        "model_name": model_name,
        "metric": "inner_product",
        "normalized": True,
        "dimension": dim,
        "num_vectors": int(index.ntotal),
        "chunk_source": str(chunk_path(split).relative_to(PROJECT_ROOT)),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"FAISS index: {faiss_path.relative_to(PROJECT_ROOT)} ({index.ntotal:,} vectors, dim={dim})")
    print(f"Metadata: {meta_path.relative_to(PROJECT_ROOT)}")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    return faiss_path, meta_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="validation",
        help="Which chunk split to index (default: validation).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()

    splits = ("train", "validation") if args.split == "all" else (args.split,)
    for split in splits:
        build_index(split, model_name=args.model, batch_size=args.batch_size)
        print()


if __name__ == "__main__":
    main()
