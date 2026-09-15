"""Semantic search over a built HotpotQA FAISS index."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.load_hotpotqa import PROJECT_ROOT
from src.retrieval.build_index import DEFAULT_MODEL, index_paths


@dataclass
class SearchHit:
    score: float
    chunk: dict


class FaissRetriever:
    """Load a FAISS index + chunk metadata and run similarity search."""

    def __init__(self, split: str = "validation", model_name: str | None = None):
        faiss_path, meta_path, config_path = index_paths(split)
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing index for split={split}. "
                f"Run: python -m src.retrieval.build_index --split {split}"
            )

        self.split = split
        self.index = faiss.read_index(str(faiss_path))
        self.chunks: list[dict] = []
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        if config_path.exists():
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = model_name or self.config.get("model_name", DEFAULT_MODEL)
        else:
            self.config = {}
            model_name = model_name or DEFAULT_MODEL

        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except (OSError, ValueError, EnvironmentError):
            self.model = SentenceTransformer(model_name)

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/metadata size mismatch: {self.index.ntotal} vs {len(self.chunks)}"
            )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        example_id: str | None = None,
    ) -> list[SearchHit]:
        """Return top-k chunks. Optionally restrict to one HotpotQA example."""
        if not query.strip():
            raise ValueError("Query must be non-empty.")

        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(emb, dtype=np.float32)

        if example_id is None:
            scores, indices = self.index.search(query_vec, top_k)
            hits: list[SearchHit] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                hits.append(SearchHit(score=float(score), chunk=self.chunks[idx]))
            return hits

        # Distractor-style search: only chunks from one example's context.
        candidate_idxs = [
            i for i, c in enumerate(self.chunks) if c["example_id"] == example_id
        ]
        if not candidate_idxs:
            raise ValueError(f"No chunks found for example_id={example_id}")

        candidate_vecs = np.vstack(
            [self.index.reconstruct(int(i)) for i in candidate_idxs]
        ).astype(np.float32)
        scores = candidate_vecs @ query_vec[0]
        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(score=float(scores[j]), chunk=self.chunks[candidate_idxs[j]])
            for j in order
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Which index to query (default: validation).",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return (default: 5).",
    )
    parser.add_argument(
        "--example-id",
        default=None,
        help="Optional: restrict search to one HotpotQA example's context.",
    )
    args = parser.parse_args()

    retriever = FaissRetriever(split=args.split)
    hits = retriever.search(
        args.query,
        top_k=args.top_k,
        example_id=args.example_id,
    )

    print(f"Split: {args.split}")
    print(f"Model: {retriever.model_name}")
    print(f"Query: {args.query}")
    if args.example_id:
        print(f"Example filter: {args.example_id}")
    print(f"Hits: {len(hits)}\n")

    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(f"{rank}. score={hit.score:.4f}  title={c['title']}  sent_id={c['sent_id']}")
        print(f"   example_id={c['example_id']}")
        print(f"   text={c['text'][:220]}")
        print()


if __name__ == "__main__":
    main()
