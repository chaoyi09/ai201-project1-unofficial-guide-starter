"""Semantic retrieval over the neu_reviews ChromaDB collection.

Public API:
    retrieve(query: str, k: int = 5) -> list[dict]

Each result dict has:
    text       — the chunk text
    source     — source filename
    professor  — professor name (or "" if not detected)
    course     — course code (or "" if not detected)
    chunk_index
    distance   — cosine distance (lower = more similar; ~0 identical, ~1 unrelated)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "neu_reviews"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 5


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"{CHROMA_DIR} not found — run `python embed.py` first"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    if not query.strip():
        raise ValueError("query must be non-empty")

    query_embedding = _model().encode([query], convert_to_numpy=True).tolist()
    res = _collection().query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    # ChromaDB returns parallel arrays nested one level (one query at a time)
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "professor": meta.get("professor", ""),
            "course": meta.get("course", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": float(dist),
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def _demo() -> None:
    """Run 3 of the 5 planning.md evaluation questions and print top-5 chunks."""
    queries = [
        "What do students say about Prof. Iraklis Tsekourakis's exams in CS5800?",
        "How does Prof. Mahsa Derakhshan grade exams in CS5800?",
        "What are the main criticisms of Prof. Shesh in CS5010?",
    ]
    for q in queries:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        hits = retrieve(q, k=5)
        for i, h in enumerate(hits, 1):
            flag = "✓" if h["distance"] < 0.5 else "✗"
            print(
                f"\n[{i}] {flag} distance={h['distance']:.4f}  "
                f"source={h['source']}  professor={h['professor']!r}  "
                f"course={h['course']!r}  idx={h['chunk_index']}"
            )
            print(f"    {h['text'][:240]}{'...' if len(h['text']) > 240 else ''}")
        print()


if __name__ == "__main__":
    _demo()
